# ---- config ----
CC      := cc
CFLAGS  := -std=c11 -Wall -Wextra -g -I.
LDFLAGS := -lm                 # link the math library (expf in sigmoid/softmax)

# library source files
SRC := src/tensor.c src/ops.c src/nn.c
OBJ := $(SRC:.c=.o)            

# ---- targets ----
.PHONY: all test memcheck clean

# default: build the demo
all: mlp

# link the demo against the compiled library objects
mlp: examples/mlp.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o examples/mlp $(LDFLAGS)

# pattern rule: any .c -> .o, rebuild if the header changes
%.o: %.c tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

# build and run the test
test: tests/test_tensor.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o tests/test_tensor $(LDFLAGS)
	./tests/test_tensor

# same as test, but with sanitizers on to catch memory errors / UB
memcheck: CFLAGS += -fsanitize=address,undefined
memcheck: clean test

clean:
	rm -f $(OBJ) examples/mlp tests/test_tensor
