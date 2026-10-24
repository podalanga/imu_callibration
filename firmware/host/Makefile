CFLAGS = -O2 -std=c99 -Wall -Wextra -Wdouble-promotion
all: host_replay host_fixed

host_replay: host_replay.c ../main/eskf.c ../main/nav.c ../main/eskf.h ../main/nav.h
	$(CC) $(CFLAGS) -Wno-double-promotion -o $@ host_replay.c ../main/eskf.c ../main/nav.c -lm

host_fixed: host_fixed.c ../main/mahony_q.c ../main/mahony_q.h
	$(CC) $(CFLAGS) -Wno-double-promotion -o $@ host_fixed.c ../main/mahony_q.c

warnings:   # the embedded sources alone, with the strict flags used on the ESP32
	$(CC) $(CFLAGS) -c ../main/eskf.c ../main/nav.c ../main/mahony_q.c -o /dev/null 2>&1 || true
	$(CC) $(CFLAGS) -fsyntax-only ../main/eskf.c ../main/nav.c ../main/mahony_q.c

clean:
	rm -f host_replay host_fixed
