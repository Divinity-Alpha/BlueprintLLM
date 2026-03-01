#pragma once

#include "Modules/ModuleManager.h"

class FBlueprintLLMModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

private:
	void RegisterMenus();
	void OnImportDSLClicked();

	TSharedPtr<class FUICommandList> PluginCommands;
};
