BLUEPRINT: BP_Demobp_Thirdpersoncontroller
PARENT: Actor

GRAPH: EventGraph

NODE n1: Event_BeginPlay
NODE n2: AddMappingContext
NODE n3: Sequence
NODE n4: IsLocalPlayerController
NODE n5: Branch
NODE n6: AddToPlayerScreen
NODE n7: AddMappingContext
NODE n8: GetVar
NODE n9: GetPlatformName
