#include "BlueprintBuilder.h"

#include "Engine/Blueprint.h"
#include "Engine/BlueprintGeneratedClass.h"
#include "Kismet2/BlueprintEditorUtils.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "K2Node_Event.h"
#include "K2Node_CustomEvent.h"
#include "K2Node_CallFunction.h"
#include "K2Node_IfThenElse.h"
#include "K2Node_ExecutionSequence.h"
#include "K2Node_ForLoop.h"
#include "K2Node_ForEachElementInEnum.h"
#include "K2Node_InputAction.h"
#include "K2Node_DynamicCast.h"
#include "K2Node_VariableGet.h"
#include "K2Node_VariableSet.h"
#include "K2Node_SpawnActorFromClass.h"
#include "K2Node_SwitchInteger.h"
#include "K2Node_SwitchString.h"
#include "K2Node_BreakStruct.h"
#include "EdGraphSchema_K2.h"
#include "Factories/BlueprintFactory.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "UObject/SavePackage.h"
#include "GameFramework/Actor.h"
#include "GameFramework/Character.h"
#include "GameFramework/Pawn.h"

UBlueprint* FBlueprintBuilder::CreateBlueprint(const FDSLBlueprint& DSL, const FString& PackagePath)
{
	// 1. Resolve parent class
	UClass* ParentClass = FindParentClass(DSL.ParentClass);
	if (!ParentClass)
	{
		UE_LOG(LogTemp, Error, TEXT("BlueprintLLM: Unknown parent class: %s, defaulting to AActor"), *DSL.ParentClass);
		ParentClass = AActor::StaticClass();
	}

	// 2. Create package
	const FString AssetName = DSL.Name.IsEmpty() ? TEXT("BP_Generated") : DSL.Name;
	const FString FullPath = PackagePath / AssetName;
	UPackage* Package = CreatePackage(*FullPath);
	if (!Package)
	{
		UE_LOG(LogTemp, Error, TEXT("BlueprintLLM: Failed to create package: %s"), *FullPath);
		return nullptr;
	}

	// 3. Create Blueprint via factory
	UBlueprintFactory* Factory = NewObject<UBlueprintFactory>();
	Factory->ParentClass = ParentClass;

	UBlueprint* Blueprint = Cast<UBlueprint>(
		Factory->FactoryCreateNew(
			UBlueprint::StaticClass(),
			Package,
			FName(*AssetName),
			RF_Public | RF_Standalone,
			nullptr,
			GWarn
		)
	);

	if (!Blueprint)
	{
		UE_LOG(LogTemp, Error, TEXT("BlueprintLLM: Failed to create Blueprint"));
		return nullptr;
	}

	// 4. Create variables
	CreateBlueprintVariables(Blueprint, DSL.Variables);

	// 5. Get or create EventGraph
	UEdGraph* EventGraph = FBlueprintEditorUtils::FindEventGraph(Blueprint);
	if (!EventGraph)
	{
		UE_LOG(LogTemp, Error, TEXT("BlueprintLLM: No EventGraph found"));
		return nullptr;
	}

	// 6. Create all nodes
	TMap<FString, UEdGraphNode*> NodeMap;
	for (const FDSLNode& NodeDef : DSL.Nodes)
	{
		UK2Node* NewNode = CreateNodeFromDef(Blueprint, EventGraph, NodeDef);
		if (NewNode)
		{
			SetNodePosition(NewNode, NodeDef.Position);
			NodeMap.Add(NodeDef.ID, NewNode);
			UE_LOG(LogTemp, Verbose, TEXT("BlueprintLLM: Created node %s (%s)"), *NodeDef.ID, *NodeDef.DSLType);
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("BlueprintLLM: Failed to create node %s (%s / %s)"),
				*NodeDef.ID, *NodeDef.DSLType, *NodeDef.UEClass);
		}
	}

	// 7. Wire connections
	ConnectPins(NodeMap, DSL.Connections);

	// 8. Compile
	FBlueprintEditorUtils::MarkBlueprintAsModified(Blueprint);
	FKismetEditorUtilities::CompileBlueprint(Blueprint);

	// 9. Save
	Package->MarkPackageDirty();
	FAssetRegistryModule::AssetCreated(Blueprint);

	const FString PackageFilename = FPackageName::LongPackageNameToFilename(FullPath, FPackageName::GetAssetPackageExtension());
	FSavePackageArgs SaveArgs;
	SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
	UPackage::SavePackage(Package, Blueprint, *PackageFilename, SaveArgs);

	UE_LOG(LogTemp, Log, TEXT("BlueprintLLM: Successfully created %s with %d nodes"),
		*AssetName, NodeMap.Num());

	return Blueprint;
}

// ============================================================
// Node creation dispatch
// ============================================================

UK2Node* FBlueprintBuilder::CreateNodeFromDef(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	const FString& UEClass = NodeDef.UEClass;

	if (UEClass == TEXT("UK2Node_Event"))
	{
		return CreateEventNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_CustomEvent"))
	{
		return CreateCustomEventNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_CallFunction"))
	{
		return CreateCallFunctionNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_IfThenElse"))
	{
		return CreateBranchNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_ExecutionSequence"))
	{
		return CreateSequenceNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_ForLoop") || UEClass == TEXT("UK2Node_ForEachLoop") || UEClass == TEXT("UK2Node_WhileLoop"))
	{
		return CreateLoopNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_DynamicCast"))
	{
		return CreateCastNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_VariableGet") || UEClass == TEXT("UK2Node_VariableSet"))
	{
		return CreateVariableNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_SpawnActorFromClass"))
	{
		return CreateSpawnActorNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_SwitchInteger") || UEClass == TEXT("UK2Node_SwitchString"))
	{
		return CreateSwitchNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_BreakStruct"))
	{
		return CreateBreakStructNode(BP, Graph, NodeDef);
	}
	if (UEClass == TEXT("UK2Node_InputAction"))
	{
		// Input action events
		UK2Node_InputAction* Node = NewObject<UK2Node_InputAction>(Graph);
		if (NodeDef.Params.Contains(TEXT("ActionName")))
		{
			Node->InputActionName = FName(*NodeDef.Params[TEXT("ActionName")]);
		}
		Node->AllocateDefaultPins();
		Graph->AddNode(Node, false, false);
		return Node;
	}

	// Flow control nodes (FlipFlop, DoOnce, Gate, MultiGate)
	if (UEClass == TEXT("UK2Node_FlipFlop") || UEClass == TEXT("UK2Node_DoOnce") ||
		UEClass == TEXT("UK2Node_Gate") || UEClass == TEXT("UK2Node_MultiGate"))
	{
		return CreateFlowControlNode(BP, Graph, NodeDef);
	}

	UE_LOG(LogTemp, Warning, TEXT("BlueprintLLM: Unhandled UE class: %s"), *UEClass);
	return nullptr;
}

// ============================================================
// Individual node creators
// ============================================================

UK2Node* FBlueprintBuilder::CreateEventNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	// Check if this event already exists in the graph (BeginPlay, Tick, etc.)
	FName EventName = FName(*NodeDef.UEEvent);

	for (UEdGraphNode* ExistingNode : Graph->Nodes)
	{
		UK2Node_Event* ExistingEvent = Cast<UK2Node_Event>(ExistingNode);
		if (ExistingEvent && ExistingEvent->EventReference.GetMemberName() == EventName)
		{
			return ExistingEvent;
		}
	}

	// Create new event node
	UK2Node_Event* EventNode = NewObject<UK2Node_Event>(Graph);
	UClass* ParentClass = BP->ParentClass;

	// Find the function in the parent class
	UFunction* EventFunc = ParentClass->FindFunctionByName(EventName);
	if (EventFunc)
	{
		EventNode->EventReference.SetFromField<UFunction>(EventFunc, false);
		EventNode->bOverrideFunction = true;
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("BlueprintLLM: Event function not found: %s"), *NodeDef.UEEvent);
	}

	EventNode->AllocateDefaultPins();
	Graph->AddNode(EventNode, false, false);

	return EventNode;
}

UK2Node* FBlueprintBuilder::CreateCustomEventNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	UK2Node_CustomEvent* Node = NewObject<UK2Node_CustomEvent>(Graph);

	FString EventName = TEXT("CustomEvent");
	if (!NodeDef.ParamKey.IsEmpty() && NodeDef.Params.Contains(NodeDef.ParamKey))
	{
		EventName = NodeDef.Params[NodeDef.ParamKey];
	}
	else if (NodeDef.Params.Contains(TEXT("EventName")))
	{
		EventName = NodeDef.Params[TEXT("EventName")];
	}

	Node->CustomFunctionName = FName(*EventName);
	Node->AllocateDefaultPins();
	Graph->AddNode(Node, false, false);

	return Node;
}

UK2Node* FBlueprintBuilder::CreateCallFunctionNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	UK2Node_CallFunction* FuncNode = NewObject<UK2Node_CallFunction>(Graph);

	// Find the UFunction from the path
	UFunction* Func = FindFunctionByPath(NodeDef.UEFunction);
	if (Func)
	{
		FuncNode->SetFromFunction(Func);
	}
	else
	{
		// Try setting by member reference
		FString ClassName, FuncName;
		if (NodeDef.UEFunction.Split(TEXT(":"), &ClassName, &FuncName))
		{
			FuncNode->FunctionReference.SetExternalMember(FName(*FuncName), nullptr);
		}
		UE_LOG(LogTemp, Warning, TEXT("BlueprintLLM: Function not found: %s"), *NodeDef.UEFunction);
	}

	FuncNode->AllocateDefaultPins();
	Graph->AddNode(FuncNode, false, false);

	// Set parameter defaults
	for (const auto& Param : NodeDef.Params)
	{
		SetPinDefaultValue(FuncNode, Param.Key, Param.Value);
	}

	return FuncNode;
}

UK2Node* FBlueprintBuilder::CreateBranchNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	UK2Node_IfThenElse* Node = NewObject<UK2Node_IfThenElse>(Graph);
	Node->AllocateDefaultPins();
	Graph->AddNode(Node, false, false);
	return Node;
}

UK2Node* FBlueprintBuilder::CreateSequenceNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	UK2Node_ExecutionSequence* Node = NewObject<UK2Node_ExecutionSequence>(Graph);
	Node->AllocateDefaultPins();
	Graph->AddNode(Node, false, false);
	return Node;
}

UK2Node* FBlueprintBuilder::CreateFlowControlNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	// These are all implemented as CallFunction nodes to macro library functions
	UK2Node_CallFunction* Node = NewObject<UK2Node_CallFunction>(Graph);

	// FlipFlop, DoOnce, Gate, MultiGate are macro library functions
	FString MacroName;
	if (NodeDef.UEClass == TEXT("UK2Node_FlipFlop")) MacroName = TEXT("FlipFlop");
	else if (NodeDef.UEClass == TEXT("UK2Node_DoOnce")) MacroName = TEXT("DoOnce");
	else if (NodeDef.UEClass == TEXT("UK2Node_Gate")) MacroName = TEXT("Gate");
	else if (NodeDef.UEClass == TEXT("UK2Node_MultiGate")) MacroName = TEXT("MultiGate");

	// These live in KismetSystemLibrary or are standard macros
	UFunction* Func = UKismetSystemLibrary::StaticClass()->FindFunctionByName(FName(*MacroName));
	if (Func)
	{
		Node->SetFromFunction(Func);
	}

	Node->AllocateDefaultPins();
	Graph->AddNode(Node, false, false);
	return Node;
}

UK2Node* FBlueprintBuilder::CreateLoopNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	if (NodeDef.UEClass == TEXT("UK2Node_ForLoop"))
	{
		UK2Node_CallFunction* Node = NewObject<UK2Node_CallFunction>(Graph);
		UFunction* Func = UKismetSystemLibrary::StaticClass()->FindFunctionByName(TEXT("ForLoop"));
		if (Func) Node->SetFromFunction(Func);
		Node->AllocateDefaultPins();
		Graph->AddNode(Node, false, false);

		// Set loop bounds from params
		if (NodeDef.Params.Contains(TEXT("FirstIndex")))
			SetPinDefaultValue(Node, TEXT("FirstIndex"), NodeDef.Params[TEXT("FirstIndex")]);
		if (NodeDef.Params.Contains(TEXT("LastIndex")))
			SetPinDefaultValue(Node, TEXT("LastIndex"), NodeDef.Params[TEXT("LastIndex")]);

		return Node;
	}

	// ForEachLoop, WhileLoop — similar pattern
	UK2Node_CallFunction* Node = NewObject<UK2Node_CallFunction>(Graph);
	Node->AllocateDefaultPins();
	Graph->AddNode(Node, false, false);
	return Node;
}

UK2Node* FBlueprintBuilder::CreateCastNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	UK2Node_DynamicCast* CastNode = NewObject<UK2Node_DynamicCast>(Graph);

	// Resolve target class from cast_class path
	UClass* TargetClass = nullptr;
	if (NodeDef.CastClass.StartsWith(TEXT("/Script/")))
	{
		TargetClass = FindObject<UClass>(nullptr, *NodeDef.CastClass);
	}
	if (!TargetClass)
	{
		// Try common class names
		if (NodeDef.CastClass.Contains(TEXT("Character"))) TargetClass = ACharacter::StaticClass();
		else if (NodeDef.CastClass.Contains(TEXT("Pawn"))) TargetClass = APawn::StaticClass();
		else TargetClass = AActor::StaticClass();
	}

	CastNode->TargetType = TargetClass;
	CastNode->AllocateDefaultPins();
	Graph->AddNode(CastNode, false, false);

	return CastNode;
}

UK2Node* FBlueprintBuilder::CreateVariableNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	// Get the variable name from params
	FString VarName;
	if (!NodeDef.ParamKey.IsEmpty() && NodeDef.Params.Contains(NodeDef.ParamKey))
	{
		VarName = NodeDef.Params[NodeDef.ParamKey];
	}

	if (VarName.IsEmpty())
	{
		UE_LOG(LogTemp, Warning, TEXT("BlueprintLLM: Variable node %s has no variable name"), *NodeDef.ID);
		return nullptr;
	}

	FName VarFName = FName(*VarName);

	if (NodeDef.UEClass == TEXT("UK2Node_VariableGet"))
	{
		UK2Node_VariableGet* Node = NewObject<UK2Node_VariableGet>(Graph);
		Node->VariableReference.SetSelfMember(VarFName);
		Node->AllocateDefaultPins();
		Graph->AddNode(Node, false, false);
		return Node;
	}
	else // UK2Node_VariableSet
	{
		UK2Node_VariableSet* Node = NewObject<UK2Node_VariableSet>(Graph);
		Node->VariableReference.SetSelfMember(VarFName);
		Node->AllocateDefaultPins();
		Graph->AddNode(Node, false, false);
		return Node;
	}
}

UK2Node* FBlueprintBuilder::CreateSpawnActorNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	UK2Node_SpawnActorFromClass* Node = NewObject<UK2Node_SpawnActorFromClass>(Graph);
	Node->AllocateDefaultPins();
	Graph->AddNode(Node, false, false);
	return Node;
}

UK2Node* FBlueprintBuilder::CreateSwitchNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	if (NodeDef.UEClass == TEXT("UK2Node_SwitchInteger"))
	{
		UK2Node_SwitchInteger* Node = NewObject<UK2Node_SwitchInteger>(Graph);
		Node->AllocateDefaultPins();
		Graph->AddNode(Node, false, false);
		return Node;
	}
	else
	{
		UK2Node_SwitchString* Node = NewObject<UK2Node_SwitchString>(Graph);
		Node->AllocateDefaultPins();
		Graph->AddNode(Node, false, false);
		return Node;
	}
}

UK2Node* FBlueprintBuilder::CreateBreakStructNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef)
{
	UK2Node_BreakStruct* Node = NewObject<UK2Node_BreakStruct>(Graph);
	// TODO: Set the struct type based on DSL type (HitResult, Vector, etc.)
	Node->AllocateDefaultPins();
	Graph->AddNode(Node, false, false);
	return Node;
}

// ============================================================
// Variable creation
// ============================================================

void FBlueprintBuilder::CreateBlueprintVariables(UBlueprint* BP, const TArray<FDSLVariable>& Variables)
{
	for (const FDSLVariable& Var : Variables)
	{
		FEdGraphPinType PinType = GetPinTypeFromString(Var.Type);
		const bool bSuccess = FBlueprintEditorUtils::AddMemberVariable(BP, FName(*Var.Name), PinType);

		if (bSuccess && !Var.DefaultValue.IsEmpty())
		{
			// Set default value
			FProperty* Prop = BP->GeneratedClass->FindPropertyByName(FName(*Var.Name));
			if (Prop)
			{
				UE_LOG(LogTemp, Verbose, TEXT("BlueprintLLM: Created variable %s : %s = %s"),
					*Var.Name, *Var.Type, *Var.DefaultValue);
			}
		}
	}
}

// ============================================================
// Connection wiring
// ============================================================

bool FBlueprintBuilder::ConnectPins(
	const TMap<FString, UEdGraphNode*>& NodeMap,
	const TArray<FDSLConnection>& Connections)
{
	int32 SuccessCount = 0;
	int32 FailCount = 0;

	for (const FDSLConnection& Conn : Connections)
	{
		// Skip literal data connections for now
		if (Conn.Type == TEXT("data_literal"))
		{
			continue;
		}

		UEdGraphNode* const* SrcNodePtr = NodeMap.Find(Conn.SourceNode);
		UEdGraphNode* const* DstNodePtr = NodeMap.Find(Conn.TargetNode);

		if (!SrcNodePtr || !DstNodePtr)
		{
			UE_LOG(LogTemp, Warning, TEXT("BlueprintLLM: Connection refs unknown node: %s -> %s"),
				*Conn.SourceNode, *Conn.TargetNode);
			FailCount++;
			continue;
		}

		UEdGraphNode* SrcNode = *SrcNodePtr;
		UEdGraphNode* DstNode = *DstNodePtr;

		// Find pins
		UEdGraphPin* SrcPin = nullptr;
		UEdGraphPin* DstPin = nullptr;

		// Try exact name match first
		SrcPin = SrcNode->FindPin(FName(*Conn.SourcePin));
		DstPin = DstNode->FindPin(FName(*Conn.TargetPin));

		// Fallback: try common pin name mappings
		if (!SrcPin)
		{
			// "Then" is often the exec output on events and functions
			if (Conn.SourcePin == TEXT("Then"))
			{
				SrcPin = SrcNode->FindPin(UEdGraphSchema_K2::PN_Then);
			}
			else if (Conn.SourcePin == TEXT("ReturnValue"))
			{
				SrcPin = SrcNode->FindPin(UEdGraphSchema_K2::PN_ReturnValue);
			}
		}

		if (!DstPin)
		{
			if (Conn.TargetPin == TEXT("Execute"))
			{
				DstPin = DstNode->FindPin(UEdGraphSchema_K2::PN_Execute);
			}
			else if (Conn.TargetPin == TEXT("Condition"))
			{
				DstPin = DstNode->FindPin(TEXT("Condition"));
			}
		}

		if (SrcPin && DstPin)
		{
			bool bConnected = SrcPin->MakeLinkTo(DstPin);
			if (bConnected)
			{
				SuccessCount++;
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("BlueprintLLM: MakeLinkTo failed: %s.%s -> %s.%s"),
					*Conn.SourceNode, *Conn.SourcePin, *Conn.TargetNode, *Conn.TargetPin);
				FailCount++;
			}
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("BlueprintLLM: Pin not found: %s.%s(%s) -> %s.%s(%s)"),
				*Conn.SourceNode, *Conn.SourcePin, SrcPin ? TEXT("OK") : TEXT("MISSING"),
				*Conn.TargetNode, *Conn.TargetPin, DstPin ? TEXT("OK") : TEXT("MISSING"));
			FailCount++;
		}
	}

	UE_LOG(LogTemp, Log, TEXT("BlueprintLLM: Wired %d/%d connections (%d failed)"),
		SuccessCount, SuccessCount + FailCount, FailCount);

	return FailCount == 0;
}

// ============================================================
// Helpers
// ============================================================

UClass* FBlueprintBuilder::FindParentClass(const FString& ClassName)
{
	if (ClassName == TEXT("Actor") || ClassName == TEXT("AActor")) return AActor::StaticClass();
	if (ClassName == TEXT("Character") || ClassName == TEXT("ACharacter")) return ACharacter::StaticClass();
	if (ClassName == TEXT("Pawn") || ClassName == TEXT("APawn")) return APawn::StaticClass();

	// Try finding by path
	UClass* Found = FindObject<UClass>(nullptr, *ClassName);
	return Found;
}

UFunction* FBlueprintBuilder::FindFunctionByPath(const FString& FunctionPath)
{
	// Format: "/Script/Engine.KismetSystemLibrary:PrintString"
	FString ClassPath, FuncName;
	if (!FunctionPath.Split(TEXT(":"), &ClassPath, &FuncName))
	{
		return nullptr;
	}

	UClass* OwnerClass = FindObject<UClass>(nullptr, *ClassPath);
	if (!OwnerClass)
	{
		// Try loading
		OwnerClass = LoadObject<UClass>(nullptr, *ClassPath);
	}

	if (OwnerClass)
	{
		return OwnerClass->FindFunctionByName(FName(*FuncName));
	}

	return nullptr;
}

FEdGraphPinType FBlueprintBuilder::GetPinTypeFromString(const FString& TypeName)
{
	FEdGraphPinType PinType;

	if (TypeName == TEXT("Int") || TypeName == TEXT("int"))
	{
		PinType.PinCategory = UEdGraphSchema_K2::PC_Int;
	}
	else if (TypeName == TEXT("Float") || TypeName == TEXT("float"))
	{
		PinType.PinCategory = UEdGraphSchema_K2::PC_Real;
		PinType.PinSubCategory = UEdGraphSchema_K2::PC_Float;
	}
	else if (TypeName == TEXT("Bool") || TypeName == TEXT("bool"))
	{
		PinType.PinCategory = UEdGraphSchema_K2::PC_Boolean;
	}
	else if (TypeName == TEXT("String") || TypeName == TEXT("string"))
	{
		PinType.PinCategory = UEdGraphSchema_K2::PC_String;
	}
	else if (TypeName == TEXT("Vector") || TypeName == TEXT("vector"))
	{
		PinType.PinCategory = UEdGraphSchema_K2::PC_Struct;
		PinType.PinSubCategoryObject = TBaseStructure<FVector>::Get();
	}
	else if (TypeName == TEXT("Rotator") || TypeName == TEXT("rotator"))
	{
		PinType.PinCategory = UEdGraphSchema_K2::PC_Struct;
		PinType.PinSubCategoryObject = TBaseStructure<FRotator>::Get();
	}
	else if (TypeName == TEXT("Object") || TypeName == TEXT("object") ||
			 TypeName == TEXT("Actor"))
	{
		PinType.PinCategory = UEdGraphSchema_K2::PC_Object;
		PinType.PinSubCategoryObject = AActor::StaticClass();
	}
	else
	{
		// Default to wildcard
		PinType.PinCategory = UEdGraphSchema_K2::PC_Wildcard;
	}

	return PinType;
}

void FBlueprintBuilder::SetNodePosition(UEdGraphNode* Node, const FVector2D& Position)
{
	if (Node)
	{
		Node->NodePosX = Position.X;
		Node->NodePosY = Position.Y;
	}
}

void FBlueprintBuilder::SetPinDefaultValue(UEdGraphNode* Node, const FString& PinName, const FString& Value)
{
	if (!Node) return;

	UEdGraphPin* Pin = Node->FindPin(FName(*PinName));
	if (Pin)
	{
		Pin->DefaultValue = Value;
	}
}
