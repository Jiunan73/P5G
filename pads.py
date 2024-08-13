import pyads
import csv



# 定義Python結構來匹配PLC中的結構

structure_def = (
   ('PositionNo', pyads.PLCTYPE_WORD, 10000),
   ('PositionX', pyads.PLCTYPE_DINT, 10000),
   ('PositionY', pyads.PLCTYPE_DINT, 10000),
   ('PositionZ', pyads.PLCTYPE_DINT, 10000),
   ('PositionNotice', pyads.PLCTYPE_WORD, 10000)
)
# 連接到PLC並打開連接
# 連接到PLC並打開連接
plc = pyads.Connection('192.168.43.110.1.1', pyads.PORT_TC3PLC1)
plc.open()

# 讀取結構數據

CarMap=plc.read_structure_by_name("GVL.PositionForAGVC", structure_def)

# 讀取CSV文件
with open('map.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# 將CSV數據填入CarMap
# 將CarMap的內容都清成0
for i in range(len(CarMap['PositionNo'])):
    CarMap['PositionNo'][i] = 0
    CarMap['PositionX'][i] = 0
    CarMap['PositionY'][i] = 0
    CarMap['PositionZ'][i] = 0
    CarMap['PositionNotice'][i] = 0

for i in range(len(data)):
    CarMap['PositionNo'][i] = int(data[i][0])
    CarMap['PositionX'][i] = int(data[i][1])
    CarMap['PositionY'][i] = int(data[i][2])    
    CarMap['PositionZ'][i] = int(data[i][3])
    CarMap['PositionNotice'][i] = int(data[i][4])


# 寫入結構數據到PLC
plc.write_structure_by_name("GVL.PositionForAGVC",CarMap, structure_def)

# 關閉連接

symbols = plc.get_all_symbols()
for i in range(len(symbols)):   
    if symbols[i].name.startswith('GVL'):
        print('\n'.join("%s: %s" % item for item in vars(symbols[i]).items()))
    

plc.close()
