import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { DriverLotsStackParamList } from "./types";
import { LotsScreen } from "../screens/driver/LotsScreen";
import { LotDetailsScreen } from "../screens/driver/LotDetailsScreen";
import { colors } from "../theme";

const Stack = createNativeStackNavigator<DriverLotsStackParamList>();

export function DriverLotsStackNavigator() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: colors.background },
        headerTintColor: colors.textPrimary,
        headerShadowVisible: false,
        contentStyle: { backgroundColor: colors.background }
      }}
    >
      <Stack.Screen name="Lots" component={LotsScreen} options={{ title: "Select Parking Lot" }} />
      <Stack.Screen
        name="LotDetails"
        component={LotDetailsScreen}
        options={{ title: "Lot Details" }}
      />
    </Stack.Navigator>
  );
}