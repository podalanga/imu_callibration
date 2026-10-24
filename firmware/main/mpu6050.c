/* MPU6050 raw-register driver: FIFO, data-ready interrupt, no DMP. ESP-IDF >= 5.2 (i2c_master.h). */
#include "mpu6050.h"
#include "driver/i2c_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#define ADDR 0x68
#define R_SMPLRT_DIV 0x19
#define R_CONFIG 0x1A
#define R_GYRO_CONFIG 0x1B
#define R_ACCEL_CONFIG 0x1C
#define R_FIFO_EN 0x23
#define R_INT_PIN_CFG 0x37
#define R_INT_ENABLE 0x38
#define R_INT_STATUS 0x3A
#define R_USER_CTRL 0x6A
#define R_PWR_MGMT_1 0x6B
#define R_FIFO_COUNTH 0x72
#define R_FIFO_R_W 0x74
#define R_WHO_AM_I 0x75

static i2c_master_bus_handle_t bus;
static i2c_master_dev_handle_t dev;

static esp_err_t wr(uint8_t reg, uint8_t val)
{
    uint8_t b[2] = {reg, val};
    return i2c_master_transmit(dev, b, 2, 50);
}

static esp_err_t rd(uint8_t reg, uint8_t *out, size_t n)
{
    return i2c_master_transmit_receive(dev, &reg, 1, out, n, 50);
}

esp_err_t mpu_init(int sda, int scl, uint32_t i2c_hz)
{
    i2c_master_bus_config_t bc = {
        .clk_source = I2C_CLK_SRC_DEFAULT, .i2c_port = I2C_NUM_0,
        .scl_io_num = scl, .sda_io_num = sda, .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,     /* GY-521 boards also have external pull-ups */
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&bc, &bus));
    i2c_device_config_t dc = { .dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = ADDR, .scl_speed_hz = i2c_hz };
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus, &dc, &dev));

    uint8_t who = 0;
    esp_err_t e = rd(R_WHO_AM_I, &who, 1);
    if (e != ESP_OK) return e;
    if (who != 0x68) ESP_LOGW("mpu", "WHO_AM_I = 0x%02x (genuine MPU6050 = 0x68); clone boards differ, check noise", who);

    wr(R_PWR_MGMT_1, 0x80);                     /* device reset */
    vTaskDelay(pdMS_TO_TICKS(100));
    wr(R_PWR_MGMT_1, 0x01);                     /* wake, PLL with X gyro reference */
    wr(R_CONFIG, 0x03);                         /* DLPF_CFG = 3: ~44 Hz, internal rate 1 kHz */
    wr(R_SMPLRT_DIV, 4);                        /* 1 kHz / (1 + 4) = 200 Hz */
    wr(R_GYRO_CONFIG, 0x08);                    /* FS_SEL = 1: +-500 dps */
    wr(R_ACCEL_CONFIG, 0x08);                   /* AFS_SEL = 1: +-4 g */
    wr(R_INT_PIN_CFG, 0x00);                    /* active high, push-pull, 50 us pulse */
    wr(R_USER_CTRL, 0x04);                      /* FIFO reset */
    wr(R_USER_CTRL, 0x40);                      /* FIFO enable */
    wr(R_FIFO_EN, 0xF8);                        /* temp, gx, gy, gz, accel into FIFO */
    wr(R_INT_ENABLE, 0x11);                     /* data ready + FIFO overflow */
    return ESP_OK;
}

esp_err_t mpu_fifo_reset(void)
{
    wr(R_USER_CTRL, 0x44);
    return wr(R_USER_CTRL, 0x40);
}

esp_err_t mpu_int_status(int *overflow)
{
    uint8_t s = 0;
    esp_err_t e = rd(R_INT_STATUS, &s, 1);
    *overflow = (s & 0x10) != 0;
    return e;
}

esp_err_t mpu_fifo_count(uint16_t *bytes)
{
    uint8_t b[2];
    esp_err_t e = rd(R_FIFO_COUNTH, b, 2);
    *bytes = (uint16_t)((b[0] << 8) | b[1]);
    return e;
}

esp_err_t mpu_fifo_read(uint8_t *buf, uint16_t bytes)
{
    return rd(R_FIFO_R_W, buf, bytes);
}

void mpu_unpack(const uint8_t *p, int16_t out[7])
{
    for (int i = 0; i < 7; i++) out[i] = (int16_t)((p[2 * i] << 8) | p[2 * i + 1]);
}

float mpu_temp_c(int16_t raw) { return (float)raw / 340.0f + 36.53f; }
