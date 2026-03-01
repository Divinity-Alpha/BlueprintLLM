#pragma once

#include "CoreMinimal.h"
#include "DSLImporter.h"

class UBlueprint;
class UEdGraph;
class UEdGraphNode;
class UK2Node;

/**
 * Creates real UBlueprint assets from parsed DSL IR.
 * Uses FBlueprintEditorUtils and UK2Node APIs.
 */
class FBlueprintBuilder
{
public:
	/**
	 * Create a Blueprint asset from a parsed DSL blueprint.
	 * @param DSL - The parsed IR
	 * @param PackagePath - Content Browser path (e.g. "/Game/BlueprintLLM/Generated")
	 * @return The created UBlueprint, or nullptr on failure
	 */
	static UBlueprint* CreateBlueprint(const FDSLBlueprint& DSL, const FString& PackagePath);

private:
	// Node creation by type
	static UK2Node* CreateEventNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateCallFunctionNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateBranchNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateSequenceNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateFlowControlNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateLoopNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateCastNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateVariableNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateCustomEventNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateSwitchNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateSpawnActorNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);
	static UK2Node* CreateBreakStructNode(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);

	// Dispatch to the right creator
	static UK2Node* CreateNodeFromDef(UBlueprint* BP, UEdGraph* Graph, const FDSLNode& NodeDef);

	// Variable management
	static void CreateBlueprintVariables(UBlueprint* BP, const TArray<FDSLVariable>& Variables);

	// Connection wiring
	static bool ConnectPins(
		const TMap<FString, UEdGraphNode*>& NodeMap,
		const TArray<FDSLConnection>& Connections);

	// Helpers
	static UClass* FindParentClass(const FString& ClassName);
	static UFunction* FindFunctionByPath(const FString& FunctionPath);
	static FEdGraphPinType GetPinTypeFromString(const FString& TypeName);
	static void SetNodePosition(UEdGraphNode* Node, const FVector2D& Position);
	static void SetPinDefaultValue(UEdGraphNode* Node, const FString& PinName, const FString& Value);
};
