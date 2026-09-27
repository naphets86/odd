# Makefile -- Subgraph-basierter Compiler+Linker (ccl.tex, Epp 2026)

CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -O2 -g
TARGET  = ccl
SRCS    = main.c compiler_full.c subgraph.c linker.c
OBJS    = $(SRCS:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Abhaengigkeiten
main.o:           main.c compiler.h linker.h subgraph.h
compiler_full.o:  compiler_full.c compiler.h subgraph.h
subgraph.o:       subgraph.c subgraph.h
linker.o:         linker.c linker.h compiler.h subgraph.h

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
