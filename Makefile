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

# build and run all tests
test: $(OBJ)
	$(CC) $(CFLAGS) tests/test_tensor.c $(OBJ) -o tests/test_tensor $(LDFLAGS)
	./tests/test_tensor
	$(CC) $(CFLAGS) tests/test_ops.c $(OBJ) -o tests/test_ops $(LDFLAGS)
	./tests/test_ops
	$(CC) $(CFLAGS) tests/test_mismatch.c $(OBJ) -o tests/test_mismatch $(LDFLAGS)
	@echo "== shape mismatch (expect error message + non-zero exit) =="
	@if ./tests/test_mismatch; then echo "[FAIL] did not exit on mismatch"; exit 1; else echo "[PASS] exited non-zero on shape mismatch"; fi

# same as test, but with sanitizers on to catch memory errors / UB
memcheck: CFLAGS += -fsanitize=address,undefined
memcheck: clean test

clean:
	rm -f $(OBJ) examples/mlp tests/test_tensor tests/test_ops tests/test_mismatch
