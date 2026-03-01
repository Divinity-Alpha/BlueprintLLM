#include "DSLImporter.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

bool FDSLImporter::ParseIR(const FString& JsonPath, FDSLBlueprint& OutBlueprint)
{
	FString JsonString;
	if (!FFileHelper::LoadFileToString(JsonString, *JsonPath))
	{
		UE_LOG(LogTemp, Error, TEXT("BlueprintLLM: Failed to read file: %s"), *JsonPath);
		return false;
	}

	return ParseIRFromString(JsonString, OutBlueprint);
}

bool FDSLImporter::ParseIRFromString(const FString& JsonString, FDSLBlueprint& OutBlueprint)
{
	TSharedPtr<FJsonObject> RootObj;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);

	if (!FJsonSerializer::Deserialize(Reader, RootObj) || !RootObj.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("BlueprintLLM: Invalid JSON"));
		return false;
	}

	// Parse metadata
	const TSharedPtr<FJsonObject>* MetaObj;
	if (RootObj->TryGetObjectField(TEXT("metadata"), MetaObj))
	{
		OutBlueprint.Name = (*MetaObj)->GetStringField(TEXT("name"));
		OutBlueprint.ParentClass = (*MetaObj)->GetStringField(TEXT("parent_class"));
		(*MetaObj)->TryGetStringField(TEXT("category"), OutBlueprint.Category);
	}

	// Parse variables
	const TArray<TSharedPtr<FJsonValue>>* VarsArray;
	if (RootObj->TryGetArrayField(TEXT("variables"), VarsArray))
	{
		for (const auto& VarVal : *VarsArray)
		{
			FDSLVariable Var;
			if (ParseVariable(VarVal->AsObject(), Var))
			{
				OutBlueprint.Variables.Add(Var);
			}
		}
	}

	// Parse nodes
	const TArray<TSharedPtr<FJsonValue>>* NodesArray;
	if (RootObj->TryGetArrayField(TEXT("nodes"), NodesArray))
	{
		for (const auto& NodeVal : *NodesArray)
		{
			FDSLNode Node;
			if (ParseNode(NodeVal->AsObject(), Node))
			{
				OutBlueprint.Nodes.Add(Node);
			}
		}
	}

	// Parse connections
	const TArray<TSharedPtr<FJsonValue>>* ConnsArray;
	if (RootObj->TryGetArrayField(TEXT("connections"), ConnsArray))
	{
		for (const auto& ConnVal : *ConnsArray)
		{
			FDSLConnection Conn;
			if (ParseConnection(ConnVal->AsObject(), Conn))
			{
				OutBlueprint.Connections.Add(Conn);
			}
		}
	}

	UE_LOG(LogTemp, Log, TEXT("BlueprintLLM: Parsed '%s' - %d nodes, %d connections, %d variables"),
		*OutBlueprint.Name, OutBlueprint.Nodes.Num(),
		OutBlueprint.Connections.Num(), OutBlueprint.Variables.Num());

	return true;
}

bool FDSLImporter::ParseNode(const TSharedPtr<FJsonObject>& JsonNode, FDSLNode& OutNode)
{
	if (!JsonNode.IsValid()) return false;

	OutNode.ID = JsonNode->GetStringField(TEXT("id"));
	OutNode.DSLType = JsonNode->GetStringField(TEXT("dsl_type"));
	OutNode.UEClass = JsonNode->GetStringField(TEXT("ue_class"));

	JsonNode->TryGetStringField(TEXT("ue_function"), OutNode.UEFunction);
	JsonNode->TryGetStringField(TEXT("ue_event"), OutNode.UEEvent);
	JsonNode->TryGetStringField(TEXT("cast_class"), OutNode.CastClass);
	JsonNode->TryGetStringField(TEXT("param_key"), OutNode.ParamKey);

	// Parse params
	const TSharedPtr<FJsonObject>* ParamsObj;
	if (JsonNode->TryGetObjectField(TEXT("params"), ParamsObj))
	{
		for (const auto& Pair : (*ParamsObj)->Values)
		{
			OutNode.Params.Add(Pair.Key, Pair.Value->AsString());
		}
	}

	// Parse position
	const TArray<TSharedPtr<FJsonValue>>* PosArray;
	if (JsonNode->TryGetArrayField(TEXT("position"), PosArray) && PosArray->Num() >= 2)
	{
		OutNode.Position.X = (*PosArray)[0]->AsNumber();
		OutNode.Position.Y = (*PosArray)[1]->AsNumber();
	}

	return true;
}

bool FDSLImporter::ParseConnection(const TSharedPtr<FJsonObject>& JsonConn, FDSLConnection& OutConn)
{
	if (!JsonConn.IsValid()) return false;

	OutConn.Type = JsonConn->GetStringField(TEXT("type"));
	JsonConn->TryGetStringField(TEXT("src_node"), OutConn.SourceNode);
	JsonConn->TryGetStringField(TEXT("src_pin"), OutConn.SourcePin);
	JsonConn->TryGetStringField(TEXT("dst_node"), OutConn.TargetNode);
	JsonConn->TryGetStringField(TEXT("dst_pin"), OutConn.TargetPin);
	JsonConn->TryGetStringField(TEXT("data_type"), OutConn.DataType);

	return true;
}

bool FDSLImporter::ParseVariable(const TSharedPtr<FJsonObject>& JsonVar, FDSLVariable& OutVar)
{
	if (!JsonVar.IsValid()) return false;

	OutVar.Name = JsonVar->GetStringField(TEXT("name"));
	OutVar.Type = JsonVar->GetStringField(TEXT("type"));
	JsonVar->TryGetStringField(TEXT("default"), OutVar.DefaultValue);

	return true;
}
