import { CommonActions, useNavigation } from "@react-navigation/native";

export function useRoleSwitcher() {
  const navigation = useNavigation();

  const switchRole = () => {
    navigation.dispatch(
      CommonActions.reset({
        index: 0,
        routes: [{ name: "ChooseMode" }]
      })
    );
  };

  return {
    switchRole
  };
}
