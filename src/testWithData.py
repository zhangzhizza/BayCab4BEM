import BayCab4BEM.bayCab4BEM as bc4b
fieldDataFile = 'cal_example_field_withoutSummer.csv'
simDataFile = 'cal_example_com_withoutSummer.csv'
bc4bObj = bc4b.BC4BEM();
bc4bObj.runWithData(fieldDataFile, simDataFile)
