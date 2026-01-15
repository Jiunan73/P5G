#include <OneWire.h>
#include <ModbusRTUSlave.h>
#include <SoftwareSerial.h>
#include <DallasTemperature.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// 定義輸入引腳
int UV_sensor1 = 18; // UV 傳感器 1 引腳
int UV_sensor2 = 8;  // UV 傳感器 2 引腳
int UV_sensor3 = 3;  // UV 傳感器 3 引腳
int ERR_sensor = 12; // 錯誤傳感器引腳
int SW_6 = 4;        // 開關 6 引腳
int SW_5 = 5;        // 開關 5 引腳
int SW_4 = 6;        // 開關 4 引腳
int SW_3 = 7;        // 開關 3 引腳
int SW_2 = 15;       // 開關 2 引腳
int SW_1 = 16;       // 開關 1 引腳

#define TX_PIN 9      // 軟體串列的傳送引腳
#define RX_PIN 10     // 軟體串列的接收引腳
#define DE_RE_PIN 11  // 控制方向的引腳，用於 Modbus 通信

SoftwareSerial mySerial(RX_PIN, TX_PIN); // 創建一個軟體串口對象
ModbusRTUSlave modbus(mySerial, DE_RE_PIN); // 創建 Modbus RTU 從屬對象
uint16_t holdingRegisters[200]; // 用於存儲 Modbus 中的保持寄存器數據的陣列

OneWire oneWire1(17); // 為溫度傳感器 1 創建 OneWire 實例
DallasTemperature sensors1(&oneWire1); // 為溫度傳感器 1 創建溫度傳感器實例
OneWire oneWire2(14); // 為溫度傳感器 2 創建 OneWire 實例
DallasTemperature sensors2(&oneWire2); // 為溫度傳感器 2 創建溫度傳感器實例
OneWire oneWire3(19); // 為溫度傳感器 3 創建 OneWire 實例
DallasTemperature sensors3(&oneWire3); // 為溫度傳感器 3 創建溫度傳感器實例
OneWire oneWire4(20); // 為溫度傳感器 4 創建 OneWire 實例
DallasTemperature sensors4(&oneWire4); // 為溫度傳感器 4 創建溫度傳感器實例
OneWire oneWire5(21); // 為溫度傳感器 5 創建 OneWire 實例
DallasTemperature sensors5(&oneWire5); // 為溫度傳感器 5 創建溫度傳感器實例

unsigned long previousMillis = 0; // 儲存上次執行的毫秒數
const long interval = 250; // 定義 LED 燈閃爍的間隔時間（毫秒）

// 用於儲存從傳感器讀取的溫度值的變數
float temp1 = 0.0;
float temp2 = 0.0;
float temp3 = 0.0;
float temp4 = 0.0;
float temp5 = 0.0;

// 計數器，從5000到8000之間計數
int heartbeatCounter = 5000;

// 用於保持 ERR 值的變數
uint16_t persistentERRValue = 0; // 用於保持 ERR 的持久值
volatile bool isTemperatureTaskHealthy = true; // 健康標誌

// 讀取溫度的任務函數
void readTemperatureTask(void *pvParameters) {
    while (true) {
        // 向每個傳感器發送請求以獲取溫度
        sensors1.requestTemperatures();
        sensors2.requestTemperatures();
        sensors3.requestTemperatures();
        sensors4.requestTemperatures();
        sensors5.requestTemperatures();

        // 獲取每個傳感器的溫度值
        temp1 = sensors1.getTempCByIndex(0);
        temp2 = sensors2.getTempCByIndex(0);
        temp3 = sensors3.getTempCByIndex(0);
        temp4 = sensors4.getTempCByIndex(0);
        temp5 = sensors5.getTempCByIndex(0);
        
        // 將溫度值存入保持寄存器（乘以 100 用於整數存儲）
        holdingRegisters[0] = (temp1 >= 0) ? (int)(temp1 * 100) : 0; // 存儲 T1
        holdingRegisters[1] = (temp2 >= 0) ? (int)(temp2 * 100) : 0; // 存儲 T2
        holdingRegisters[2] = (temp3 >= 0) ? (int)(temp3 * 100) : 0; // 存儲 T3
        holdingRegisters[3] = (temp4 >= 0) ? (int)(temp4 * 100) : 0; // 存儲 T4
        holdingRegisters[4] = (temp5 >= 0) ? (int)(temp5 * 100) : 0; // 存儲 T5

        isTemperatureTaskHealthy = true; // 任務完成標記健康

        vTaskDelay(800 / portTICK_PERIOD_MS);  // 延遲 800 毫秒
    }
}

// 監控任務函數
void watchdogTask(void *pvParameters) {
    while (true) {
        // 檢查健康標誌
        if (!isTemperatureTaskHealthy) {
            Serial.println("Watchdog triggered. Restarting due to temperature task failure...");
            esp_restart();
        }
        vTaskDelay(5000 / portTICK_PERIOD_MS); // 每 5 秒檢查一次
    }
}

void setup() {
    // 設置引腳模式
    pinMode(LED_BUILTIN, OUTPUT); 
    pinMode(UV_sensor1, INPUT);
    pinMode(UV_sensor2, INPUT);
    pinMode(UV_sensor3, INPUT);
    pinMode(ERR_sensor, INPUT);
    pinMode(SW_1, INPUT);
    pinMode(SW_2, INPUT);
    pinMode(SW_3, INPUT);
    pinMode(SW_4, INPUT);
    pinMode(SW_5, INPUT);
    pinMode(SW_6, INPUT);

    // 初始化溫度傳感器
    sensors1.begin();
    sensors2.begin();
    sensors3.begin();
    sensors4.begin();
    sensors5.begin();
  
    // 初始化串口通信
    Serial.begin(9600);
    // 配置 Modbus 的保持寄存器
    modbus.configureHoldingRegisters(holdingRegisters, 200);
    mySerial.begin(38400); // 啟動軟體串口，設置波特率為 38400

    // 讀取開關狀態
    int SW1 = analogRead(SW_1);
    int SW2 = analogRead(SW_2);
    int SW3 = analogRead(SW_3);
    int SW4 = analogRead(SW_4);
    int SW5 = analogRead(SW_5);
    int SW6 = analogRead(SW_6);

    // 根據開關狀態計算二進制值
    int binaryValue = (SW1 < 2095 ? 1 : 0) |
                      (SW2 < 2095 ? 1 : 0) << 1 |
                      (SW3 < 2095 ? 1 : 0) << 2 |
                      (SW4 < 2095 ? 1 : 0) << 3 |
                      (SW5 < 2095 ? 1 : 0) << 4 |
                      (SW6 < 2095 ? 1 : 0) << 5;
                    
    // 開始 Modbus 通信，傳入計算出的二進制值
    modbus.begin(binaryValue, 38400, SERIAL_8N1);
    Serial.print("Number Binary Value: "); Serial.println(binaryValue);

    // 創建讀取溫度的任務（運行在核心 0）
    xTaskCreatePinnedToCore(
        readTemperatureTask,   // 任務函數
        "TempTask",            // 任務名稱
        2048,                  // 堆疊大小
        NULL,                  // 傳遞的參數
        1,                     // 優先級
        NULL,                  // 任務句柄
        0                      // 綁定的核心 ID（0）
    );

    // 創建監控任務
    xTaskCreate(watchdogTask, "Watchdog Task", 2048, NULL, 1, NULL);
}

void loop() {
    modbus.poll(); // 處理 Modbus 請求

    // LED 控制邏輯
    unsigned long currentMillis = millis(); // 獲取當前毫秒
    if (currentMillis - previousMillis >= interval) { // 檢查是否達到閾值
        previousMillis = currentMillis; // 更新上次執行的時間
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); // 切換 LED 狀態
    }
    
    if (holdingRegisters[100] == 1) {
        esp_restart(); // 如果條件滿足，則重啟 ESP
    }

    // 計數器
    heartbeatCounter++;
    // 如果計數器達到 8000，則重置為 5000
    if (heartbeatCounter > 8000) {
        heartbeatCounter = 5000;
    }

    // 讀取 UV 傳感器和錯誤傳感器的值
    int UV1 = analogRead(UV_sensor1);
    int UV2 = analogRead(UV_sensor2);
    int UV3 = analogRead(UV_sensor3);
    int ERR = analogRead(ERR_sensor);

    // 從 ERR 傳感器讀取數值
    if (ERR == 4095) {
        persistentERRValue = ERR; // 保持 ERR 的值
        holdingRegisters[8] = persistentERRValue; // 將持久的 ERR 寫入保持寄存器
    } else {
        holdingRegisters[8] = heartbeatCounter; // 否則，輸出 heartbeatCounter
    }

    // 更新保持寄存器中的感測器值
    holdingRegisters[5] = UV1; // 存儲 UV 傳感器 1 數據
    holdingRegisters[6] = UV2; // 存儲 UV 傳感器 2 數據
    holdingRegisters[7] = UV3; // 存儲 UV 傳感器 3 數據
    holdingRegisters[9] = holdingRegisters[101]; // 更新其他寄存器值

    // 將傳感器數據輸出到串口
    Serial.print("[0]T1: "); Serial.println(holdingRegisters[0]); 
    Serial.print("[1]T2: "); Serial.println(holdingRegisters[1]); 
    Serial.print("[2]T3: "); Serial.println(holdingRegisters[2]); 
    Serial.print("[3]T4: "); Serial.println(holdingRegisters[3]); 
    Serial.print("[4]T5: "); Serial.println(holdingRegisters[4]); 
    Serial.print("[5]UV1: "); Serial.println(holdingRegisters[5]); 
    Serial.print("[6]UV2: "); Serial.println(holdingRegisters[6]); 
    Serial.print("[7]UV3: "); Serial.println(holdingRegisters[7]); 
    Serial.print("[8]ERR HB: "); Serial.println(holdingRegisters[8]); 
    Serial.print("[9]heartbeatout: "); Serial.println(holdingRegisters[9]);
    Serial.print("[100]Rest: "); Serial.println(holdingRegisters[100]);
    Serial.print("[101]heartbeatin: "); Serial.println(holdingRegisters[101]);
}
