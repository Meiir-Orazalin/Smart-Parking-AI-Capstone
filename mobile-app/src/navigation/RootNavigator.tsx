import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { ChooseModeScreen } from "../screens/ChooseModeScreen";
import { DriverTabsNavigator } from "./DriverTabs";
import { OperationsTabsNavigator } from "./OperationsTabs";
import { RootStackParamList } from "./types";

const Stack = createNativeStackNavigator<RootStackParamList>();

export function RootNavigator() {
  return (
    <Stack.Navigator
      initialRouteName="ChooseMode"
      screenOptions={{ headerShown: false, animation: "fade" }}
    >
      <Stack.Screen name="ChooseMode" component={ChooseModeScreen} />
      <Stack.Screen name="DriverMode" component={DriverTabsNavigator} />
      <Stack.Screen name="OperationsMode" component={OperationsTabsNavigator} />
    </Stack.Navigator>
  );
}
