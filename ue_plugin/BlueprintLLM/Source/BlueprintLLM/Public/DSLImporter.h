#pragma once

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"

// Represents a single node from the IR
struct FDSLNode
{
	FString ID;
	FString DSLType;
	FString UEClass;
	FString UEFunction;  // For CallFunction nodes
	FString UEEvent;     // For Event nodes
	FString CastClass;   // For DynamicCast nodes
	FString ParamKey;    // For Variable/CustomEvent nodes
	TMap<FString, FString> Params;
	FVector2D Position;
};

// Represents a connection between pins
struct FDSLConnection
{
	FString Type;  // "exec" or "data"
	FString SourceNode;
	FString SourcePin;
	FString TargetNode;
	FString TargetPin;
	FString DataType;  // For data connections
};

// Represents a variable declaration
struct FDSLVariable
{
	FString Name;
	FString Type;
	FString DefaultValue;
};

// Complete parsed blueprint IR
struct FDSLBlueprint
{
	FString Name;
	FString ParentClass;
	FString Category;
	TArray<FDSLVariable> Variables;
	TArray<FDSLNode> Nodes;
	TArray<FDSLConnection> Connections;
};

/**
 * Reads .blueprint.json IR files produced by the Python DSL parser.
 */
class FDSLImporter
{
public:
	/**
	 * Parse a .blueprint.json file into an FDSLBlueprint.
	 * @param JsonPath - Full path to the .blueprint.json file
	 * @param OutBlueprint - The parsed result
	 * @return true if parsing succeeded
	 */
	static bool ParseIR(const FString& JsonPath, FDSLBlueprint& OutBlueprint);

	/**
	 * Parse a JSON string directly.
	 */
	static bool ParseIRFromString(const FString& JsonString, FDSLBlueprint& OutBlueprint);

private:
	static bool ParseNode(const TSharedPtr<FJsonObject>& JsonNode, FDSLNode& OutNode);
	static bool ParseConnection(const TSharedPtr<FJsonObject>& JsonConn, FDSLConnection& OutConn);
	static bool ParseVariable(const TSharedPtr<FJsonObject>& JsonVar, FDSLVariable& OutVar);
};
