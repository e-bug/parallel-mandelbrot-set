CC=gcc
LD=${CC}
override CFLAGS+= -Wall -Werror -DDEBUG
LDFLAGS+=-lm

all: naive_escape_time naive_smooth_escape_time \
     opti1_escape_time opti2_escape_time opti3_escape_time 

naive_escape_time: palettes.c io_binary.c naive_escape_time.c
        $(LD) $? $(CFLAGS) $(LDFLAGS) -o $@

naive_smooth_escape_time: palettes.c io_binary.c naive_smooth_escape_time.c
        $(LD) $? $(CFLAGS) $(LDFLAGS) -o $@

opti1_escape_time: palettes.c io_binary.c opti1_escape_time.c
        $(LD) $? $(CFLAGS) $(LDFLAGS) -o $@

opti2_escape_time: palettes.c io_binary.c opti2_escape_time.c
        $(LD) $? $(CFLAGS) $(LDFLAGS) -o $@

opti3_escape_time: palettes.c io_binary.c opti3_escape_time.c
        $(LD) $? $(CFLAGS) $(LDFLAGS) -o $@

