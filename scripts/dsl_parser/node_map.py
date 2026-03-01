"""
BlueprintLLM Node Mapping Table
Maps DSL node type names to UE5 class/function paths.
"""

NODE_MAP = {
    # ---- EVENTS ----
    "Event_BeginPlay": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveBeginPlay", "exec_out": ["Then"]},
    "Event_Tick": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveTick", "exec_out": ["Then"], "data_out": {"DeltaSeconds": "float"}},
    "Event_EndPlay": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveEndPlay", "exec_out": ["Then"]},
    "Event_ActorBeginOverlap": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveActorBeginOverlap", "exec_out": ["Then"], "data_out": {"OtherActor": "object"}},
    "Event_ActorEndOverlap": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveActorEndOverlap", "exec_out": ["Then"], "data_out": {"OtherActor": "object"}},
    "Event_AnyDamage": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveAnyDamage", "exec_out": ["Then"], "data_out": {"Damage": "float"}},
    "Event_Hit": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveHit", "exec_out": ["Then"]},
    "Event_InputAction": {"ue_class": "UK2Node_InputAction", "param_key": "ActionName", "exec_out": ["Pressed", "Released"]},
    "Event_CustomEvent": {"ue_class": "UK2Node_CustomEvent", "param_key": "EventName", "exec_out": ["Then"]},
    "Event_Custom": {"ue_class": "UK2Node_CustomEvent", "param_key": "EventName", "exec_out": ["Then"]},
    "Event_Unknown": {"ue_class": "UK2Node_Event", "ue_event": "Unknown", "exec_out": ["Then"]},
    "Event_EndOverlap": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveActorEndOverlap", "exec_out": ["Then"], "data_out": {"OtherActor": "object"}},

    # ---- FLOW CONTROL ----
    "Branch": {"ue_class": "UK2Node_IfThenElse", "exec_in": ["Execute"], "data_in": {"Condition": "bool"}, "exec_out": ["True", "False"]},
    "Sequence": {"ue_class": "UK2Node_ExecutionSequence", "exec_in": ["Execute"], "exec_out": ["A", "B", "C", "D", "E", "F"]},
    "FlipFlop": {"ue_class": "UK2Node_FlipFlop", "exec_in": ["Execute"], "exec_out": ["A", "B"], "data_out": {"IsA": "bool"}},
    "DoOnce": {"ue_class": "UK2Node_DoOnce", "exec_in": ["Execute", "Reset"], "exec_out": ["Completed"]},
    "Gate": {"ue_class": "UK2Node_Gate", "exec_in": ["Enter", "Open", "Close", "Toggle"], "exec_out": ["Exit"]},
    "MultiGate": {"ue_class": "UK2Node_MultiGate", "exec_in": ["Execute"], "exec_out": ["Out_0", "Out_1", "Out_2", "Out_3"]},

    # ---- LOOPS ----
    "ForLoop": {"ue_class": "UK2Node_ForLoop", "exec_in": ["Execute"], "data_in": {"FirstIndex": "int", "LastIndex": "int"}, "exec_out": ["LoopBody", "Completed"], "data_out": {"Index": "int"}},
    "ForEachLoop": {"ue_class": "UK2Node_ForEachLoop", "exec_in": ["Execute"], "data_in": {"Array": "array"}, "exec_out": ["LoopBody", "Completed"], "data_out": {"Element": "object", "Index": "int"}},
    "WhileLoop": {"ue_class": "UK2Node_WhileLoop", "exec_in": ["Execute"], "data_in": {"Condition": "bool"}, "exec_out": ["LoopBody", "Completed"]},

    # ---- CASTS ----
    "CastToCharacter": {"ue_class": "UK2Node_DynamicCast", "cast_class": "/Script/Engine.Character", "exec_in": ["Execute"], "data_in": {"Object": "object"}, "exec_out": ["CastSucceeded", "CastFailed"], "data_out": {"AsCharacter": "object"}},
    "CastToPawn": {"ue_class": "UK2Node_DynamicCast", "cast_class": "/Script/Engine.Pawn", "exec_in": ["Execute"], "data_in": {"Object": "object"}, "exec_out": ["CastSucceeded", "CastFailed"], "data_out": {"AsPawn": "object"}},
    "CastToPlayerController": {"ue_class": "UK2Node_DynamicCast", "cast_class": "/Script/Engine.PlayerController", "exec_in": ["Execute"], "data_in": {"Object": "object"}, "exec_out": ["CastSucceeded", "CastFailed"]},

    # ---- VARIABLES ----
    "VariableGet": {"ue_class": "UK2Node_VariableGet", "param_key": "VarName", "data_out": {"Value": "wildcard"}},
    "GetVar": {"ue_class": "UK2Node_VariableGet", "param_key": "Variable", "data_out": {"Value": "wildcard"}},
    "VariableSet": {"ue_class": "UK2Node_VariableSet", "param_key": "VarName", "exec_in": ["Execute"], "data_in": {"Value": "wildcard"}, "exec_out": ["Then"]},
    "SetVar": {"ue_class": "UK2Node_VariableSet", "param_key": "Variable", "exec_in": ["Execute"], "data_in": {"Value": "wildcard"}, "exec_out": ["Then"]},

    # ---- SWITCH ----
    "SwitchOnInt": {"ue_class": "UK2Node_SwitchInteger", "exec_in": ["Execute"], "data_in": {"Selection": "int"}, "exec_out": ["Default"], "dynamic_pins": True},
    "SwitchOnString": {"ue_class": "UK2Node_SwitchString", "exec_in": ["Execute"], "data_in": {"Selection": "string"}, "exec_out": ["Default"], "dynamic_pins": True},

    # ---- KISMET SYSTEM ----
    "PrintString": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:PrintString", "exec_in": ["Execute"], "data_in": {"InString": "string"}, "exec_out": ["Then"]},
    "Delay": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:Delay", "exec_in": ["Execute"], "data_in": {"Duration": "float"}, "exec_out": ["Completed"]},
    "IsValid": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:IsValid", "data_in": {"Input": "object"}, "data_out": {"ReturnValue": "bool"}},
    "DestroyActor": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:K2_DestroyActor", "exec_in": ["Execute"], "exec_out": ["Then"]},
    "SetTimerByFunctionName": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:K2_SetTimerDelegate", "exec_in": ["Execute"], "data_in": {"FunctionName": "string", "Time": "float", "Looping": "bool"}, "exec_out": ["Then"]},
    "ClearTimerByFunctionName": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:K2_ClearTimerDelegate", "exec_in": ["Execute"], "data_in": {"FunctionName": "string"}, "exec_out": ["Then"]},
    "ResetDoOnce": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:ResetDoOnce", "exec_in": ["Execute"], "exec_out": ["Then"]},

    # ---- ACTOR ----
    "GetActorLocation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:K2_GetActorLocation", "data_out": {"ReturnValue": "vector"}},
    "SetActorLocation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:K2_SetActorLocation", "exec_in": ["Execute"], "data_in": {"NewLocation": "vector"}, "exec_out": ["Then"]},
    "AddActorLocalRotation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:K2_AddActorLocalRotation", "exec_in": ["Execute"], "data_in": {"DeltaRotation": "rotator"}, "exec_out": ["Then"]},
    "AddActorLocalOffset": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:K2_AddActorLocalOffset", "exec_in": ["Execute"], "data_in": {"DeltaLocation": "vector"}, "exec_out": ["Then"]},
    "GetActorForwardVector": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:GetActorForwardVector", "data_out": {"ReturnValue": "vector"}},
    "GetActorRotation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:K2_GetActorRotation", "data_out": {"ReturnValue": "rotator"}},
    "GetDistanceTo": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:GetDistanceTo", "data_in": {"OtherActor": "object"}, "data_out": {"ReturnValue": "float"}},
    "SetActorHiddenInGame": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:SetActorHiddenInGame", "exec_in": ["Execute"], "data_in": {"bNewHidden": "bool"}, "exec_out": ["Then"]},
    "TeleportTo": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:K2_TeleportTo", "exec_in": ["Execute"], "data_in": {"DestLocation": "vector"}, "exec_out": ["Then"]},
    "SetVisibility": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.SceneComponent:SetVisibility", "exec_in": ["Execute"], "data_in": {"bNewVisibility": "bool"}, "exec_out": ["Then"]},

    # ---- GAMEPLAY STATICS ----
    "GetPlayerPawn": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.GameplayStatics:GetPlayerPawn", "data_out": {"ReturnValue": "object"}},
    "GetWorldDeltaSeconds": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.GameplayStatics:GetWorldDeltaSeconds", "data_out": {"ReturnValue": "float"}},
    "PlaySoundAtLocation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.GameplayStatics:PlaySoundAtLocation", "exec_in": ["Execute"], "data_in": {"Sound": "object", "Location": "vector"}, "exec_out": ["Then"]},
    "SpawnActorFromClass": {"ue_class": "UK2Node_SpawnActorFromClass", "exec_in": ["Execute"], "data_in": {"ActorClass": "class", "SpawnTransform": "transform"}, "exec_out": ["Then"], "data_out": {"ReturnValue": "object"}},

    # ---- PHYSICS ----
    "SetSimulatePhysics": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.PrimitiveComponent:SetSimulatePhysics", "exec_in": ["Execute"], "data_in": {"bSimulate": "bool"}, "exec_out": ["Then"]},
    "AddImpulse": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.PrimitiveComponent:AddImpulse", "exec_in": ["Execute"], "data_in": {"Impulse": "vector"}, "exec_out": ["Then"]},

    # ---- MATH (FLOAT) ----
    "AddFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "SubtractFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Subtract_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "MultiplyFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Multiply_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "DivideFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Divide_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "ClampFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:FClamp", "data_in": {"Value": "float", "Min": "float", "Max": "float"}, "data_out": {"ReturnValue": "float"}},
    "RandomFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:RandomFloat", "data_out": {"ReturnValue": "float"}},
    "RandomFloatInRange": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:RandomFloatInRange", "data_in": {"Min": "float", "Max": "float"}, "data_out": {"ReturnValue": "float"}},
    "SelectFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:SelectFloat", "data_in": {"A": "float", "B": "float", "Select": "bool"}, "data_out": {"ReturnValue": "float"}},

    # ---- MATH (INT) ----
    "AddInt": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_IntInt", "data_in": {"A": "int", "B": "int"}, "data_out": {"ReturnValue": "int"}},
    "SubtractInt": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Subtract_IntInt", "data_in": {"A": "int", "B": "int"}, "data_out": {"ReturnValue": "int"}},
    "RandomInteger": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:RandomInteger", "data_in": {"Max": "int"}, "data_out": {"ReturnValue": "int"}},
    "Modulo": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Percent_IntInt", "data_in": {"A": "int", "B": "int"}, "data_out": {"ReturnValue": "int"}},

    # ---- MATH (COMPARISON) ----
    "LessThan": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Less_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},
    "GreaterThan": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Greater_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},
    "LessEqual": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:LessEqual_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},
    "LessEqualFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:LessEqual_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},
    "EqualEqual": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:EqualEqual_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},

    # ---- VECTOR / ROTATOR ----
    "MakeVector": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:MakeVector", "data_in": {"X": "float", "Y": "float", "Z": "float"}, "data_out": {"ReturnValue": "vector"}},
    "MakeRotator": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:MakeRotator", "data_in": {"Roll": "float", "Pitch": "float", "Yaw": "float"}, "data_out": {"ReturnValue": "rotator"}},
    "VectorLerp": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:VLerp", "data_in": {"A": "vector", "B": "vector", "Alpha": "float"}, "data_out": {"ReturnValue": "vector"}},
    "VectorDistance": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Vector_Distance", "data_in": {"A": "vector", "B": "vector"}, "data_out": {"ReturnValue": "float"}},
    "VSize": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:VSize", "data_in": {"A": "vector"}, "data_out": {"ReturnValue": "float"}},

    # ---- STRING ----
    "Concatenate": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetStringLibrary:Concat_StrStr", "data_in": {"A": "string", "B": "string"}, "data_out": {"ReturnValue": "string"}},
    "GetDisplayName": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:GetDisplayName", "data_in": {"Object": "object"}, "data_out": {"ReturnValue": "string"}},
    "GetLength": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetStringLibrary:Len", "data_in": {"S": "string"}, "data_out": {"ReturnValue": "int"}},

    # ---- UI/WIDGET ----
    "CreateWidget": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/UMG.WidgetBlueprintLibrary:Create", "exec_in": ["Execute"], "data_in": {"WidgetClass": "class"}, "exec_out": ["Then"], "data_out": {"ReturnValue": "object"}},
    "AddToViewport": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/UMG.UserWidget:AddToViewport", "exec_in": ["Execute"], "data_in": {"Target": "object"}, "exec_out": ["Then"]},
    "RemoveFromParent": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/UMG.UserWidget:RemoveFromParent", "exec_in": ["Execute"], "exec_out": ["Then"]},

    # ---- ARRAY ----
    "ArrayLength": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Length", "data_in": {"Target": "array"}, "data_out": {"ReturnValue": "int"}},
    "GetArrayLength": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Length", "data_in": {"Target": "array"}, "data_out": {"ReturnValue": "int"}},
    "GetArraySize": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Length", "data_in": {"Target": "array"}, "data_out": {"ReturnValue": "int"}},
    "Contains": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Contains", "data_in": {"Target": "array", "Item": "wildcard"}, "data_out": {"ReturnValue": "bool"}},
    "ClearArray": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Clear", "exec_in": ["Execute"], "data_in": {"Target": "array"}, "exec_out": ["Then"]},
    "Get": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Get", "data_in": {"Target": "array", "Index": "int"}, "data_out": {"ReturnValue": "wildcard"}},
    "GetArrayItemAtIndex": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Get", "data_in": {"Target": "array", "Index": "int"}, "data_out": {"ReturnValue": "wildcard"}},
    "RemoveAt": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_RemoveItem", "exec_in": ["Execute"], "data_in": {"Target": "array", "Index": "int"}, "exec_out": ["Then"]},
    "AddUnique": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_AddUnique", "exec_in": ["Execute"], "data_in": {"Target": "array", "Item": "wildcard"}, "exec_out": ["Then"]},

    # ---- MISC ----
    "LineTraceSingle": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:LineTraceSingle", "exec_in": ["Execute"], "data_in": {"Start": "vector", "End": "vector"}, "exec_out": ["Then"], "data_out": {"ReturnValue": "bool"}},
    "BreakHitResult": {"ue_class": "UK2Node_BreakStruct", "data_in": {"HitResult": "hitresult"}, "data_out": {"Location": "vector", "Normal": "vector"}},
    "GetWorld": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:GetWorld", "data_out": {"ReturnValue": "object"}},
    "OpenGate": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:OpenGate", "exec_in": ["Execute"], "exec_out": ["Then"]},
    # ---- MATH (additional comparisons) ----
    "GreaterEqualFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:GreaterEqual_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},
    "GreaterEqualInt": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:GreaterEqual_IntInt", "data_in": {"A": "int", "B": "int"}, "data_out": {"ReturnValue": "bool"}},
    "GreaterThanFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Greater_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},
    "LessThanFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Less_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},
    "GreaterEqual": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:GreaterEqual_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},
    "NotEqual": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:NotEqual_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},
    "NotEqualEqual": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:NotEqual_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "bool"}},

    # ---- MATH (boolean) ----
    "Not": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Not_PreBool", "data_in": {"A": "bool"}, "data_out": {"ReturnValue": "bool"}},
    "NotBool": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Not_PreBool", "data_in": {"A": "bool"}, "data_out": {"ReturnValue": "bool"}},
    "NOT": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Not_PreBool", "data_in": {"A": "bool"}, "data_out": {"ReturnValue": "bool"}},
    "And": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:BooleanAND", "data_in": {"A": "bool", "B": "bool"}, "data_out": {"ReturnValue": "bool"}},
    "BooleanAND": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:BooleanAND", "data_in": {"A": "bool", "B": "bool"}, "data_out": {"ReturnValue": "bool"}},
    "Or": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:BooleanOR", "data_in": {"A": "bool", "B": "bool"}, "data_out": {"ReturnValue": "bool"}},
    "BooleanOR": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:BooleanOR", "data_in": {"A": "bool", "B": "bool"}, "data_out": {"ReturnValue": "bool"}},

    # ---- MATH (trig/minmax) ----
    "Sin": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Sin", "data_in": {"A": "float"}, "data_out": {"ReturnValue": "float"}},
    "Max": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:FMax", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "MaxFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:FMax", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "Min": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:FMin", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "Lerp": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Lerp", "data_in": {"A": "float", "B": "float", "Alpha": "float"}, "data_out": {"ReturnValue": "float"}},
    "IncInteger": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_IntInt", "data_in": {"A": "int", "B": "int"}, "data_out": {"ReturnValue": "int"}},
    "GetRandomInt": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:RandomInteger", "data_in": {"Max": "int"}, "data_out": {"ReturnValue": "int"}},

    # ---- VECTOR (additional) ----
    "BreakVector": {"ue_class": "UK2Node_BreakStruct", "data_in": {"Vector": "vector"}, "data_out": {"X": "float", "Y": "float", "Z": "float"}},
    "SubtractVector": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Subtract_VectorVector", "data_in": {"A": "vector", "B": "vector"}, "data_out": {"ReturnValue": "vector"}},
    "VectorMultiply": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Multiply_VectorFloat", "data_in": {"A": "vector", "B": "float"}, "data_out": {"ReturnValue": "vector"}},
    "VectorLength": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:VSize", "data_in": {"A": "vector"}, "data_out": {"ReturnValue": "float"}},
    "AddVector": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_VectorVector", "data_in": {"A": "vector", "B": "vector"}, "data_out": {"ReturnValue": "vector"}},
    "AddVectors": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_VectorVector", "data_in": {"A": "vector", "B": "vector"}, "data_out": {"ReturnValue": "vector"}},
    "Vector": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:MakeVector", "data_in": {"X": "float", "Y": "float", "Z": "float"}, "data_out": {"ReturnValue": "vector"}},
    "VectorVariable": {"ue_class": "UK2Node_VariableGet", "param_key": "Variable", "data_out": {"Value": "vector"}},
    "Vector_Lerp": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:VLerp", "data_in": {"A": "vector", "B": "vector", "Alpha": "float"}, "data_out": {"ReturnValue": "vector"}},
    "Vector_Distance": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Vector_Distance", "data_in": {"A": "vector", "B": "vector"}, "data_out": {"ReturnValue": "float"}},

    # ---- ACTOR (additional) ----
    "SetActorRotation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:K2_SetActorRotation", "exec_in": ["Execute"], "data_in": {"NewRotation": "rotator"}, "exec_out": ["Then"]},
    "GetWorldLocation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.SceneComponent:K2_GetComponentLocation", "data_out": {"ReturnValue": "vector"}},
    "GetWorldRotation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.SceneComponent:K2_GetComponentRotation", "data_out": {"ReturnValue": "rotator"}},
    "GetForwardVector": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:GetActorForwardVector", "data_out": {"ReturnValue": "vector"}},
    "ActorHasTag": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:ActorHasTag", "data_in": {"Tag": "name"}, "data_out": {"ReturnValue": "bool"}},
    "GetActorHasTag": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:ActorHasTag", "data_in": {"Tag": "name"}, "data_out": {"ReturnValue": "bool"}},
    "GetActorTags": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:GetActorTags", "data_out": {"ReturnValue": "array"}},
    "ToggleVisibility": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.SceneComponent:ToggleVisibility", "exec_in": ["Execute"], "exec_out": ["Then"]},
    "RotateActorOnAxis": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:K2_AddActorLocalRotation", "exec_in": ["Execute"], "data_in": {"DeltaRotation": "rotator"}, "exec_out": ["Then"]},
    "Event_Overlap": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveActorBeginOverlap", "exec_out": ["Then"], "data_out": {"OtherActor": "object"}},
    "Event_OverlapActor": {"ue_class": "UK2Node_Event", "ue_event": "ReceiveActorBeginOverlap", "exec_out": ["Then"], "data_out": {"OtherActor": "object"}},

    # ---- PHYSICS (additional) ----
    "AddForce": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.PrimitiveComponent:AddForce", "exec_in": ["Execute"], "data_in": {"Force": "vector"}, "exec_out": ["Then"]},
    "AddForceAtLocation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.PrimitiveComponent:AddForceAtLocation", "exec_in": ["Execute"], "data_in": {"Force": "vector", "Location": "vector"}, "exec_out": ["Then"]},
    "AddImpulseAtLocation": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.PrimitiveComponent:AddImpulseAtLocation", "exec_in": ["Execute"], "data_in": {"Impulse": "vector", "Location": "vector"}, "exec_out": ["Then"]},
    "AddDownwardForce": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.PrimitiveComponent:AddForce", "exec_in": ["Execute"], "data_in": {"Force": "vector"}, "exec_out": ["Then"]},

    # ---- MOVEMENT / INPUT (additional) ----
    "AddMovementInput": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Pawn:AddMovementInput", "exec_in": ["Execute"], "data_in": {"WorldDirection": "vector", "ScaleValue": "float"}, "exec_out": ["Then"]},
    "GetInputAxisValue": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:GetInputAxisValue", "data_in": {"AxisName": "name"}, "data_out": {"ReturnValue": "float"}},
    "MoveTo": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/AIModule.AIBlueprintHelperLibrary:SimpleMoveToLocation", "exec_in": ["Execute"], "data_in": {"Goal": "vector"}, "exec_out": ["Then"]},

    # ---- GAMEPLAY (additional) ----
    "GetGameTimeInSeconds": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.GameplayStatics:GetTimeSeconds", "data_out": {"ReturnValue": "float"}},
    "GetGameTimeSeconds": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.GameplayStatics:GetTimeSeconds", "data_out": {"ReturnValue": "float"}},
    "GetPlayerCharacter": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.GameplayStatics:GetPlayerCharacter", "data_out": {"ReturnValue": "object"}},
    "GetDistanceToActor": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.Actor:GetDistanceTo", "data_in": {"OtherActor": "object"}, "data_out": {"ReturnValue": "float"}},
    "GetActorArray": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.GameplayStatics:GetAllActorsOfClass", "data_out": {"ReturnValue": "array"}},

    # ---- GATE (additional) ----
    "CloseGate": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:CloseGate", "exec_in": ["Execute"], "exec_out": ["Then"]},
    "EnterGate": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:EnterGate", "exec_in": ["Execute"], "exec_out": ["Then"]},

    # ---- STRING (additional) ----
    "IntToString": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetStringLibrary:Conv_IntToString", "data_in": {"InInt": "int"}, "data_out": {"ReturnValue": "string"}},
    "ToString": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetStringLibrary:Conv_FloatToString", "data_in": {"InFloat": "float"}, "data_out": {"ReturnValue": "string"}},
    "ConvertToText": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetStringLibrary:Conv_FloatToString", "data_in": {"InFloat": "float"}, "data_out": {"ReturnValue": "string"}},
    "AppendText": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetStringLibrary:Concat_StrStr", "data_in": {"A": "string", "B": "string"}, "data_out": {"ReturnValue": "string"}},
    "GetDisplayText": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:GetDisplayName", "data_in": {"Object": "object"}, "data_out": {"ReturnValue": "string"}},

    # ---- TIMING (additional) ----
    "RetriggerableDelay": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:RetriggerableDelay", "exec_in": ["Execute"], "data_in": {"Duration": "float"}, "exec_out": ["Completed"]},
    "TimerByFunction": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:K2_SetTimerDelegate", "exec_in": ["Execute"], "data_in": {"FunctionName": "string", "Time": "float", "Looping": "bool"}, "exec_out": ["Then"]},

    # ---- VARIABLE (additional) ----
    "GetVariable": {"ue_class": "UK2Node_VariableGet", "param_key": "Variable", "data_out": {"Value": "wildcard"}},
    "SetVariable": {"ue_class": "UK2Node_VariableSet", "param_key": "Variable", "exec_in": ["Execute"], "data_in": {"Value": "wildcard"}, "exec_out": ["Then"]},

    # ---- UI (additional) ----
    "SetText": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/UMG.TextBlock:SetText", "exec_in": ["Execute"], "data_in": {"InText": "string"}, "exec_out": ["Then"]},
    "AddToPlayerScreen": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/UMG.UserWidget:AddToViewport", "exec_in": ["Execute"], "data_in": {"Target": "object"}, "exec_out": ["Then"]},

    # ---- MISC (additional) ----
    "GetHitNormal": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.HitResult:GetHitNormal", "data_out": {"ReturnValue": "vector"}},
    "Print": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:PrintString", "exec_in": ["Execute"], "data_in": {"InString": "string"}, "exec_out": ["Then"]},
    "PrintFloat": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetSystemLibrary:PrintString", "exec_in": ["Execute"], "data_in": {"InString": "string"}, "exec_out": ["Then"]},

    # ---- ARRAY (additional) ----
    "ArrayAdd": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Add", "exec_in": ["Execute"], "data_in": {"Target": "array", "Item": "wildcard"}, "exec_out": ["Then"]},
    "ArrayGet": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Get", "data_in": {"Target": "array", "Index": "int"}, "data_out": {"ReturnValue": "wildcard"}},
    "ArrayClear": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Clear", "exec_in": ["Execute"], "data_in": {"Target": "array"}, "exec_out": ["Then"]},
    "ArrayContains": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Contains", "data_in": {"Target": "array", "Item": "wildcard"}, "data_out": {"ReturnValue": "bool"}},
    "ArrayRemove": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_RemoveItem", "exec_in": ["Execute"], "data_in": {"Target": "array", "Item": "wildcard"}, "exec_out": ["Then"]},
    "ArrayRemoveIndex": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Remove", "exec_in": ["Execute"], "data_in": {"Target": "array", "Index": "int"}, "exec_out": ["Then"]},
    "GetArrayElement": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Get", "data_in": {"Target": "array", "Index": "int"}, "data_out": {"ReturnValue": "wildcard"}},
    "GetElementAtIndex": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Get", "data_in": {"Target": "array", "Index": "int"}, "data_out": {"ReturnValue": "wildcard"}},
    "AddActor": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetArrayLibrary:Array_Add", "exec_in": ["Execute"], "data_in": {"Target": "array", "Item": "wildcard"}, "exec_out": ["Then"]},

    # ---- GAMEPLAY-SPECIFIC (model inventions) ----
    "AddToHealth": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "AddToArmor": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "AddToPlayerScore": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "AddToPlayerHealth": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "AddToPlayerArmor": {"ue_class": "UK2Node_CallFunction", "ue_function": "/Script/Engine.KismetMathLibrary:Add_FloatFloat", "data_in": {"A": "float", "B": "float"}, "data_out": {"ReturnValue": "float"}},
    "SetHealth": {"ue_class": "UK2Node_VariableSet", "param_key": "Variable", "exec_in": ["Execute"], "data_in": {"Value": "wildcard"}, "exec_out": ["Then"]},

}

# Aliases for alternate names the model sometimes uses
ALIASES = {
    "Event_EndOverlap": "Event_ActorEndOverlap",
    "Event AnyDamage": "Event_AnyDamage",
    "TimerByFunctionName": "SetTimerByFunctionName",
    "SetInt": "VariableSet",
    "AddToCollection": "AddUnique",
    "RemoveFromCollection": "RemoveAt",
    "AddToSet": "AddUnique",
    "AddToPlayerHealth": "AddFloat",
    "AddToPlayerArmor": "AddFloat",
    "AppendToArrray": "AddUnique",
    "FindFurthest": "GetDistanceTo",
    "PlaySound": "PlaySoundAtLocation",
    "CallEvent": "Event_CustomEvent",
    "CallFunction": "PrintString",
    "OnTimer": "Event_CustomEvent",
    "DistanceBetween": "VectorDistance",
    "Vector_MakeFromRotator": "MakeRotator",
    "ForLoopFirst": "ForLoop",
    "ForLoopLast": "ForLoop",
    "LoopBody": "ForLoop",
    "GetDisplayText": "GetDisplayName",
    "GetArrayElement": "GetArrayItemAtIndex",
    "AddActor": "AddUnique",
    "AddToPlayerScreen": "AddToViewport",
}


def resolve(name: str):
    """Resolve DSL node type to (canonical_name, mapping_dict_or_None)."""
    if name in NODE_MAP:
        return name, NODE_MAP[name]
    if name in ALIASES:
        c = ALIASES[name]
        return c, NODE_MAP.get(c)
    # Dynamic CastTo<X> patterns
    if name.startswith("CastTo") and name not in NODE_MAP:
        cls = name[6:]
        return name, {"ue_class": "UK2Node_DynamicCast", "cast_class": cls,
                       "exec_in": ["Execute"], "data_in": {"Object": "object"},
                       "exec_out": ["CastSucceeded", "CastFailed"],
                       "data_out": {f"As{cls}": "object"}}
    return name, None
