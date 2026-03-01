using UnrealBuildTool;

public class BlueprintLLM : ModuleRules
{
	public BlueprintLLM(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore"
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"UnrealEd",
			"BlueprintGraph",
			"KismetCompiler",
			"Kismet",
			"GraphEditor",
			"Json",
			"JsonUtilities",
			"Slate",
			"SlateCore",
			"EditorStyle",
			"ToolMenus",
			"AssetTools",
			"ContentBrowser"
		});
	}
}
