/* ESP32 (original, dual-core LX6) firmware: MPU6050 acquisition + ESKF + telemetry + HIL replay.
 *
 * Wiring (GY-521 board): VCC 3V3, GND, SDA GPIO21, SCL GPIO22, INT GPIO4.
 * Marker button: the on-board BOOT button (GPIO0, active low). Press it while stopped over a marker.
 * UART0 at 921600 baud carries binary frames (protocol.h); ESP-IDF log output is silenced after boot.
 *
 * Tasks
 *   ISR (data ready)  latch esp_timer time, notify acq task
 *   acq_task  core 1  read FIFO, reconstruct timestamps, queue samples        (live mode)
 *   rx_task   core 0  parse host frames: mode commands and replay samples   (HIL mode)
 *   nav_task  core 1  calibrate, ESKF, detector, aids; send RAW/EST/MARK/STAT frames
 */
#include <string.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "esp_log.h"
#include "mpu6050.h"
#include "nav.h"
#include "protocol.h"
#include "calib_params.h"

#define PIN_SDA 21
#define PIN_SCL 22
#define PIN_INT 4
#define PIN_BTN 0
#define UART_PORT UART_NUM_0
#define BAUD 921600
#define PERIOD_US 5000
#define G0 9.80665f
#define DEG2RAD 0.017453292519943295f

typedef struct { uint32_t seq; int64_t t_us; int16_t raw[7]; float acc[3], gyr[3]; uint8_t replay; } sample_t;

static QueueHandle_t q_samples, q_marks;
static TaskHandle_t h_acq;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static volatile int64_t isr_time_us;
static volatile uint32_t isr_count;
static volatile int replay_mode;
static volatile uint32_t st_fifo_overflows, st_queue_drops, st_seq_gaps;

static void IRAM_ATTR mpu_isr(void *arg)
{
    int64_t t = esp_timer_get_time();
    portENTER_CRITICAL_ISR(&mux);
    isr_time_us = t; isr_count++;
    portEXIT_CRITICAL_ISR(&mux);
    BaseType_t woken = pdFALSE;
    vTaskNotifyGiveFromISR(h_acq, &woken);
    portYIELD_FROM_ISR(woken);
}

static void IRAM_ATTR btn_isr(void *arg)
{
    static int64_t last = 0;
    int64_t t = esp_timer_get_time();
    if (t - last < 300000) return;              /* 300 ms debounce */
    last = t;
    BaseType_t woken = pdFALSE;
    xQueueSendFromISR(q_marks, &t, &woken);
    portYIELD_FROM_ISR(woken);
}

static void send_frame(uint8_t type, const void *payload, uint8_t len)
{
    uint8_t buf[4 + 255 + 2];
    buf[0] = SYNC0; buf[1] = SYNC1; buf[2] = type; buf[3] = len;
    memcpy(&buf[4], payload, len);
    uint16_t c = crc16_ccitt(&buf[2], len + 2);
    buf[4 + len] = (uint8_t)(c & 0xFF); buf[5 + len] = (uint8_t)(c >> 8);
    uart_write_bytes(UART_PORT, buf, (size_t)len + 6);
}

static void to_si(const int16_t r[7], float acc[3], float gyr[3])
{
    for (int i = 0; i < 3; i++) {
        acc[i] = (float)r[i] / MPU_ACC_LSB_PER_G * G0;
        gyr[i] = (float)r[4 + i] / MPU_GYRO_LSB_PER_DPS * DEG2RAD;
    }
}

static void acq_task(void *arg)
{
    static uint8_t buf[MPU_PACKET * 32];
    uint32_t seq = 0;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(50));
        if (replay_mode) continue;
        int ovf = 0;
        mpu_int_status(&ovf);
        if (ovf) { st_fifo_overflows++; mpu_fifo_reset(); continue; }
        int64_t t_last; uint32_t c0, c1; uint16_t bytes;
        do {                                      /* consistent (time, count) pair */
            portENTER_CRITICAL(&mux); c0 = isr_count; t_last = isr_time_us; portEXIT_CRITICAL(&mux);
            mpu_fifo_count(&bytes);
            portENTER_CRITICAL(&mux); c1 = isr_count; portEXIT_CRITICAL(&mux);
        } while (c0 != c1);
        uint16_t n = bytes / MPU_PACKET;
        if (n > 32) n = 32;                       /* the rest is read on the next wake-up */
        if (n == 0) continue;
        if (mpu_fifo_read(buf, n * MPU_PACKET) != ESP_OK) continue;
        for (uint16_t i = 0; i < n; i++) {
            sample_t s = {0};
            mpu_unpack(&buf[i * MPU_PACKET], s.raw);
            s.seq = seq++;
            /* newest packet in the FIFO belongs to the latest data-ready interrupt */
            s.t_us = t_last - (int64_t)(bytes / MPU_PACKET - 1 - i) * PERIOD_US;
            to_si(s.raw, s.acc, s.gyr);
            if (xQueueSend(q_samples, &s, 0) != pdTRUE) st_queue_drops++;
        }
    }
}

static void rx_task(void *arg)
{
    uint8_t b, hdr[2], pl[64], crc[2];
    uint32_t rseq = 0;
    for (;;) {
        if (uart_read_bytes(UART_PORT, &b, 1, portMAX_DELAY) != 1 || b != SYNC0) continue;
        if (uart_read_bytes(UART_PORT, &b, 1, pdMS_TO_TICKS(10)) != 1 || b != SYNC1) continue;
        if (uart_read_bytes(UART_PORT, hdr, 2, pdMS_TO_TICKS(10)) != 2 || hdr[1] > sizeof(pl)) continue;
        if (uart_read_bytes(UART_PORT, pl, hdr[1], pdMS_TO_TICKS(20)) != hdr[1]) continue;
        if (uart_read_bytes(UART_PORT, crc, 2, pdMS_TO_TICKS(10)) != 2) continue;
        uint8_t tmp[66]; tmp[0] = hdr[0]; tmp[1] = hdr[1]; memcpy(&tmp[2], pl, hdr[1]);
        if (crc16_ccitt(tmp, hdr[1] + 2) != (uint16_t)(crc[0] | (crc[1] << 8))) continue;
        if (hdr[0] == T_CTRL) {
            replay_mode = (pl[0] == CMD_REPLAY) ? 1 : (pl[0] == CMD_LIVE ? 0 : replay_mode);
            if (pl[0] != CMD_RESET) rseq = 0;
            sample_t s = {0}; s.seq = UINT32_MAX;              /* reset marker for nav_task */
            xQueueSend(q_samples, &s, portMAX_DELAY);
        } else if (hdr[0] == T_REPLAY && replay_mode) {
            pl_replay_t r; memcpy(&r, pl, sizeof r);
            sample_t s = {0};
            s.seq = rseq++; s.t_us = r.t_us; s.replay = 1;
            memcpy(s.acc, r.acc, sizeof s.acc); memcpy(s.gyr, r.gyro, sizeof s.gyr);
            xQueueSend(q_samples, &s, portMAX_DELAY);          /* host paces the stream */
        }
    }
}

static nav_t nav;

static void nav_reset(void)
{
    nav_config_t cfg; nav_calib_t cal;
    nav_default_config(&cfg);
    calib_params_load(&cal, &cfg);                /* generated from calib.yaml and noise.yaml */
    nav_init(&nav, &cfg, &cal);
}

static void nav_task(void *arg)
{
    sample_t s;
    uint32_t last_seq = 0, max_cycles = 0, samples = 0, cpu_mhz = 0;
    int have_seq = 0;
    int64_t t_stat = esp_timer_get_time();
    cpu_mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
    nav_reset();
    for (;;) {
        int64_t tm;
        while (xQueueReceive(q_marks, &tm, 0) == pdTRUE) {
            static uint16_t id = 0;
            pl_mark_t m = {.t_us = tm, .id = id++};
            send_frame(T_MARK, &m, sizeof m);
        }
        if (xQueueReceive(q_samples, &s, pdMS_TO_TICKS(100)) != pdTRUE) continue;
        if (s.seq == UINT32_MAX) { nav_reset(); have_seq = 0; continue; }
        if (have_seq && s.seq != last_seq + 1) st_seq_gaps++;
        last_seq = s.seq; have_seq = 1; samples++;

        if (!s.replay) {
            pl_raw_t r = {.seq = s.seq, .t_us = s.t_us, .ax = s.raw[0], .ay = s.raw[1], .az = s.raw[2],
                          .temp = s.raw[3], .gx = s.raw[4], .gy = s.raw[5], .gz = s.raw[6]};
            send_frame(T_RAW, &r, sizeof r);
        }
        uint32_t c0 = esp_cpu_get_cycle_count();
        int running = nav_step(&nav, s.t_us, s.acc, s.gyr);
        uint32_t cyc = esp_cpu_get_cycle_count() - c0;
        if (cyc > max_cycles) max_cycles = cyc;

        /* every sample in replay (for exact comparison), every 4th live (50 Hz) */
        if (running && (s.replay || (s.seq & 3u) == 0)) {
            pl_est_t e = {.t_us = s.t_us, .cycles = cyc, .still = (uint8_t)nav.still};
            memcpy(e.p, nav.f.p, sizeof e.p); memcpy(e.v, nav.f.v, sizeof e.v); memcpy(e.q, nav.f.q, sizeof e.q);
            memcpy(e.ba, nav.f.ba, sizeof e.ba); memcpy(e.bg, nav.f.bg, sizeof e.bg);
            send_frame(T_EST, &e, sizeof e);
        }
        if (esp_timer_get_time() - t_stat > 1000000) {
            t_stat = esp_timer_get_time();
            pl_stat_t st = {samples, st_fifo_overflows, st_queue_drops, st_seq_gaps, max_cycles,
                            nav.cnt.gated, nav.cnt.nan_resets, cpu_mhz};
            send_frame(T_STAT, &st, sizeof st);
            max_cycles = 0;
        }
    }
}

void app_main(void)
{
    uart_config_t uc = { .baud_rate = BAUD, .data_bits = UART_DATA_8_BITS, .parity = UART_PARITY_DISABLE,
                         .stop_bits = UART_STOP_BITS_1, .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
                         .source_clk = UART_SCLK_DEFAULT };
    uart_driver_install(UART_PORT, 4096, 16384, 0, NULL, 0);
    uart_param_config(UART_PORT, &uc);

    if (mpu_init(PIN_SDA, PIN_SCL, 400000) != ESP_OK) ESP_LOGE("main", "MPU6050 not found: check wiring");
    esp_log_level_set("*", ESP_LOG_NONE);        /* UART0 now carries binary frames only */

    q_samples = xQueueCreate(64, sizeof(sample_t));
    q_marks = xQueueCreate(16, sizeof(int64_t));

    xTaskCreatePinnedToCore(nav_task, "nav", 8192, NULL, 10, NULL, 1);
    xTaskCreatePinnedToCore(acq_task, "acq", 4096, NULL, 12, &h_acq, 1);
    xTaskCreatePinnedToCore(rx_task, "rx", 4096, NULL, 5, NULL, 0);

    gpio_config_t io = { .pin_bit_mask = (1ULL << PIN_INT), .mode = GPIO_MODE_INPUT, .intr_type = GPIO_INTR_POSEDGE };
    gpio_config(&io);
    gpio_config_t bt = { .pin_bit_mask = (1ULL << PIN_BTN), .mode = GPIO_MODE_INPUT, .pull_up_en = GPIO_PULLUP_ENABLE,
                         .intr_type = GPIO_INTR_NEGEDGE };
    gpio_config(&bt);
    gpio_install_isr_service(0);
    gpio_isr_handler_add(PIN_INT, mpu_isr, NULL);
    gpio_isr_handler_add(PIN_BTN, btn_isr, NULL);
    mpu_fifo_reset();
}
