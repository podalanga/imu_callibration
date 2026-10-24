idf_component_register(SRCS "main.c" "mpu6050.c" "eskf.c" "nav.c" "mahony_q.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_driver_i2c esp_driver_uart esp_driver_gpio esp_timer)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wdouble-promotion -O2)
