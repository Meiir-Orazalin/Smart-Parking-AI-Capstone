export type RootStackParamList = {
  ChooseMode: undefined;
  DriverMode: undefined;
  OperationsMode: undefined;
};

export type DriverTabParamList = {
  LotsStack: undefined;
  Profile: undefined;
};

export type DriverLotsStackParamList = {
  Lots: undefined;
  LotDetails: { lotId: string };
};

export type OperationsTabParamList = {
  Dashboard: undefined;
  Cameras: undefined;
  Layouts: undefined;
};