import React from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Ionicons } from "@expo/vector-icons";
import { DriverTabParamList } from "./types";
import { DriverLotsStackNavigator } from "./DriverLotsStack";
import { ProfileScreen } from "../screens/driver/ProfileScreen";
import { colors } from "../theme";

const Tab = createBottomTabNavigator<DriverTabParamList>();

export function DriverTabsNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          backgroundColor: colors.surface,
          borderTopColor: colors.border
        },
        tabBarActiveTintColor: colors.accent,
        tabBarInactiveTintColor: colors.textMuted,
        tabBarIcon: ({ color, size }) => {
          const iconByRoute: Record<keyof DriverTabParamList, keyof typeof Ionicons.glyphMap> = {
            LotsStack: "car-outline",
            Profile: "person-outline"
          };
          return <Ionicons name={iconByRoute[route.name]} color={color} size={size} />;
        }
      })}
    >
      <Tab.Screen name="LotsStack" component={DriverLotsStackNavigator} options={{ title: "Lots" }} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
}