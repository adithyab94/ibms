#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_log.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include <stdbool.h>
#include "esp_heap_caps.h"
#include "input.h"

// TWAI (CAN) includes
#include "driver/twai.h"
#include "nvs_flash.h"

// TensorFlow Lite Micro includes
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include timer for inference time measurement
#include "esp_timer.h"

// Model data
#include "model.h"

#define TAG "TFLITE"
// Message IDs
#define MSG_ID_CURRENT 0x119
#define MSG_ID_TEMP1_2 0x133
#define MSG_ID_TEMP3_4 0x134
#define MSG_ID_CELLV1_4 0x150
#define MSG_ID_CELLV5_8 0x151
#define MSG_ID_CELLV9_12 0x152
#define MSG_ID_SOC 0xB
#define MSG_ID_TEMP_1 0X73E

// Define the maximum expected time delta for scaling
#define MAX_TIME_DELTA 10.0 // seconds, adjust this value based on your application's requirements

namespace
{
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
    int inference_count = 0;
    constexpr int kTensorArenaSize = 1024 * 160;
    uint8_t tensor_arena[kTensorArenaSize];
    float inputs[300][4];         // Adjusted to hold 4 features
    int64_t last_sample_time = 0; // Store the timestamp of the last sample
}

void twai_receive_task(void *pvParameters);

void setup()
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Initialize TWAI driver
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(GPIO_NUM_5, GPIO_NUM_35, TWAI_MODE_NORMAL);
    twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();
    ESP_ERROR_CHECK(twai_driver_install(&g_config, &t_config, &f_config));
    ESP_ERROR_CHECK(twai_start());

    // Initialize TensorFlow Lite Micro
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "Model schema version %" PRIu32 " not supported!", model->version());
        return;
    }

    static tflite::MicroMutableOpResolver<14> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddTanh();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddTranspose();
    micro_op_resolver.AddCallOnce();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddUnpack();
    micro_op_resolver.AddPack();
    micro_op_resolver.AddSplit();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddLogistic();
    micro_op_resolver.AddReshape();

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    // Create the TWAI receive task
    xTaskCreate(twai_receive_task, "twai_receive", 4096, NULL, 5, NULL);
}

void loop()
{
    // No operations in loop
}

void twai_receive_task(void *pvParameters)
{
    float total_voltage = 0.0;
    float total_current = 0.0;
    float total_temperature = 0.0;
    int sample_count = 0;
    float voltage_samples[12] = {0};
    float temp_samples[4] = {0};
    bool flagUpdateCAN = false;

    while (1)
    {
        twai_message_t rx_msg;
        esp_err_t ret = twai_receive(&rx_msg, pdMS_TO_TICKS(2000)); // Increased timeout to 2000ms
        if (ret == ESP_OK)
        {
            // ESP_LOGI(TAG, "TWAI message received: 0x%03lx, DLC: %d", rx_msg.identifier, rx_msg.data_length_code);

            switch (rx_msg.identifier)
            {
            case MSG_ID_CURRENT:
                if (rx_msg.data_length_code == 8)
                {
                    int32_t current_raw;
                    current_raw = (int32_t)((rx_msg.data[3] << 24) | (rx_msg.data[2] << 16) | (rx_msg.data[1] << 8) | rx_msg.data[0]);

                    total_current = current_raw * 0.0009765625; // Apply correct scaling factor
                    // ESP_LOGI(TAG, "Current: %f", total_current);
                    flagUpdateCAN = true;
                }
                else
                {
                    ESP_LOGW(TAG, "Unexpected data length for CURRENT message");
                }
                break;

                // case MSG_ID_TEMP:
                // if (rx_msg.data_length_code == 8)
                // {
                //     int32_t temp_raw;
                //    temp_raw = (int32_t) ( (rx_msg.data[3] << 24) | (rx_msg.data[2] << 16) | (rx_msg.data[1] << 8) | rx_msg.data[0]);

                //     total_temperature = temp_raw * 0.0009765625; // Apply correct scaling factor
                //     // ESP_LOGI(TAG, "Current: %f", total_current);
                //     // flagUpdateCAN = true;
                // }
                // break;

            case MSG_ID_TEMP_1:
                if (rx_msg.data_length_code == 8)
                {
                    int32_t temp_raw;
                    total_temperature = (int16_t)((rx_msg.data[1] << 8) | rx_msg.data[0]);

                    // total_temperature = temp_raw * 0.0009765625; // Apply correct scaling factor
                    // ESP_LOGI(TAG, "Current: %f", total_current);
                    flagUpdateCAN = true;
                }

                else
                {
                    ESP_LOGW(TAG, "Unexpected data length for CURRENT message");
                }
                break;

            case MSG_ID_CELLV1_4:
                if (rx_msg.data_length_code == 8)
                {
                    memcpy(&voltage_samples[0], &rx_msg.data[0], sizeof(float));
                    memcpy(&voltage_samples[1], &rx_msg.data[2], sizeof(float));
                    memcpy(&voltage_samples[2], &rx_msg.data[4], sizeof(float));
                    memcpy(&voltage_samples[3], &rx_msg.data[6], sizeof(float));
                    // ESP_LOGI(TAG, "raw V byte 0: 0x%02X", voltage_samples[0]);
                    // ESP_LOGI(TAG, "raw V byte 1: 0x%02X", voltage_samples[1]);
                    // ESP_LOGI(TAG, "CellV1: %f, CellV2: %f, CellV3: %f, CellV4: %f",
                    // voltage_samples[0], voltage_samples[1], voltage_samples[2], voltage_samples[3]);

                    total_voltage = (int16_t)((rx_msg.data[1] << 8) | rx_msg.data[0]);
                    // ESP_LOGI(TAG, "total_voltage: %f", total_voltage);
                    flagUpdateCAN = true;
                }
                else
                {
                    ESP_LOGW(TAG, "Unexpected data length for CELLV1_4 message");
                }
                break;

            default:
                // ESP_LOGW(TAG, "Unknown message ID: 0x%03lx", rx_msg.identifier);
                break;
            }

            if (flagUpdateCAN)
            {

                ESP_LOGI(TAG, "Total Voltage: %f, Total Current: %f, Total Temperature: %f", total_voltage, total_current, total_temperature);

                float total_current_sc = (total_current - 43.6) / (175 - 43.6);
                float total_voltage_sc = (total_voltage) / (6000);
                float total_temperature_sc = (total_temperature - (-40)) / (55 - (-40));

                // Get the current time
                int64_t current_time = esp_timer_get_time();

                // Calculate the time delta (in microseconds) and scale it
                float delta_t = 0.0;
                if (sample_count > 0)
                {
                    delta_t = static_cast<float>(current_time - last_sample_time) / 1e6;    // Convert to seconds
                    delta_t = std::min(static_cast<float>(delta_t / MAX_TIME_DELTA), 1.0f); // Scale delta_t between 0 and 1
                }
                last_sample_time = current_time;

                // ESP_LOGI(TAG, "Total Voltage_sc: %f, Total Current_SC: %f, Total Temperature_SC: %f", total_voltage_sc, total_current_sc, total_temperature_sc);

                // Prepare input for the model
                inputs[sample_count][0] = total_voltage_sc;
                inputs[sample_count][1] = total_current_sc;
                inputs[sample_count][2] = total_temperature_sc;
                inputs[sample_count][3] = delta_t;

                sample_count++;
                flagUpdateCAN = false;
            }

            if (sample_count >= 300)
            {
                // Log indicating that 300 samples have been received
                ESP_LOGI(TAG, "300 samples received, performing inference");

                // Record the start time
                int64_t start_time = esp_timer_get_time();

                // Perform inference after collecting 300 samples
                memcpy(input->data.f, inputs, sizeof(float) * 300 * 4); // Adjust to accommodate 4 features now
                TfLiteStatus invoke_status = interpreter->Invoke();

                // Record the end time
                int64_t end_time = esp_timer_get_time();

                // Calculate and log the inference time
                int64_t inference_time = end_time - start_time;
                ESP_LOGI(TAG, "Inference time: %lld microseconds", inference_time);

                if (invoke_status != kTfLiteOk)
                {
                    ESP_LOGE(TAG, "Invoke failed");
                }
                else
                {
                    float soh = output->data.f[0];
                    soh = soh * 100;
                    // Add factor 0.01 to soc and scale it

                    ESP_LOGI(TAG, " SOH: %f", soh);

                    uint16_t soh_sc = soh / 0.01; // soc / 0.01;

                    // Send SOC over CAN
                    twai_message_t soh_msg;
                    soh_msg.identifier = MSG_ID_SOC;
                    soh_msg.extd = 0;
                    soh_msg.rtr = 0;
                    soh_msg.data_length_code = 8;
                    memcpy(soh_msg.data, &soh_sc, sizeof(uint16_t));

                    ret = twai_transmit(&soh_msg, pdMS_TO_TICKS(10));
                    if (ret != ESP_OK)
                    {
                        ESP_LOGE(TAG, "Failed to transmit SOH message: %s", esp_err_to_name(ret));
                    }

                    sample_count = 0; // Reset sample count
                }
            }
        }
        else if (ret == ESP_ERR_TIMEOUT)
        {
            ESP_LOGW(TAG, "TWAI receive timeout");
        }
        else
        {
            ESP_LOGE(TAG, "TWAI receive failed: %s", esp_err_to_name(ret));
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
