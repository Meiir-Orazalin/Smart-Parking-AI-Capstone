import React from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { OperationsTabParamList } from "./types";
import { colors } from "../theme";
import { DashboardScreen } from "../screens/operations/DashboardScreen";
import { CamerasScreen } from "../screens/operations/CamerasScreen";
import { LayoutsScreen } from "../screens/operations/LayoutsScreen";

const Tab = createBottomTabNavigator<OperationsTabParamList>();

export function OperationsTabsNavigator() {
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
          const iconByRoute: Record<
            keyof OperationsTabParamList,
            keyof typeof MaterialCommunityIcons.glyphMap
          > = {
            Dashboard: "view-dashboard-outline",
            Cameras: "cctv",
            Layouts: "map-outline"
          };
          return <MaterialCommunityIcons name={iconByRoute[route.name]} color={color} size={size} />;
        }
      })}
    >
      <Tab.Screen name="Dashboard" component={DashboardScreen} />
      <Tab.Screen name="Cameras" component={CamerasScreen} />
      <Tab.Screen name="Layouts" component={LayoutsScreen} />
    </Tab.Navigator>
  );
}
