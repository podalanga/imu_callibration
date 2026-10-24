#ifndef MPU6050_H
#define MPU6050_H
#include <stdint.h>
#include "esp_err.h"

#define MPU_PACKET 14                       /* ax ay az temp gx gy gz, big-endian int16 */
#define MPU_ACC_LSB_PER_G 8192.0f           /* +-4 g */
#define MPU_GYRO_LSB_PER_DPS 65.5f          /* +-500 dps */

esp_err_t mpu_init(int sda, int scl, uint32_t i2c_hz);
esp_err_t mpu_fifo_reset(void);
/* reads INT_STATUS (clears it); *overflow set if the FIFO overflowed */
esp_err_t mpu_int_status(int *overflow);
esp_err_t mpu_fifo_count(uint16_t *bytes);
esp_err_t mpu_fifo_read(uint8_t *buf, uint16_t bytes);
void mpu_unpack(const uint8_t *p, int16_t out[7]);
float mpu_temp_c(int16_t raw);

#endif
