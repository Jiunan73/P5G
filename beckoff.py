        PositionX
        PositionY
PositionZ
PositionTagID



PositionForAGVC                             : PointData ;
        PositionTotalFloorForAGVC            : ARRAY[1..10] OF PointData ;
        bAGVCReadPositionData                : BOOL ;
        bAGVCNeedWritePositionData       : BOOL ;
        bAGVCWritePositionDataOK          : BOOL ;

TYPE
        PointData : STRUCT
                PositionNo :ARRAY[1..10000] OF WORD ;
                PositionX :ARRAY[1..10000] OF DINT ;
                PositionY :ARRAY[1..10000] OF DINT ;
                PositionZ :ARRAY[1..10000] OF DINT ;
                PositionNotice :ARRAY[1..10000] OF WORD ;
                //TravelWayTable :ARRAY[1..20000] OF DWORD ;
        END_STRUCT
END_TYPE
