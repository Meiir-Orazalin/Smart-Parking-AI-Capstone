import React from "react";
import { NavigationContainer, DarkTheme } from "@react-navigation/native";
import { StatusBar } from "expo-status-bar";
import { RootNavigator } from "./src/navigation/RootNavigator";
import { theme } from "./src/constants/theme";
import { AppDataProvider } from "./src/hooks/AppDataProvider";

const navigationTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: theme.colors.backgroundBottom,
    card: theme.colors.card,
    border: theme.colors.border,
    text: theme.colors.textPrimary,
    primary: theme.colors.accent
  }
};

export default function App() {
  return (
    <NavigationContainer theme={navigationTheme}>
      <StatusBar style="light" />
      <AppDataProvider>
        <RootNavigator />
      </AppDataProvider>
    </NavigationContainer>
  );
}
