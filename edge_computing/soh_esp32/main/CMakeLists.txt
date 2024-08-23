
#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc main_twai.cc model.cc
    PRIV_REQUIRES spi_flash
    INCLUDE_DIRS "" "${CMAKE_CURRENT_LIST_DIR}/../managed_components/json/cJSON"
    REQUIRES fatfs esp_wifi nvs_flash json esp_timer
    PRIV_REQUIRES
    )
