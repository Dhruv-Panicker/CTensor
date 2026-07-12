# ---- config ----
CC      := cc
CFLAGS  := -std=c11 -Wall -Wextra -g -I.
LDFLAGS := -lm                 # link the math library (expf in sigmoid/softmax)

# library source files
SRC := src/tensor.c src/ops.c src/nn.c
OBJ := $(SRC:.c=.o)            

# ---- targets ----
.PHONY: all test memcheck bench clean

# default: build the demo
all: mlp

# link the demo against the compiled library objects
mlp: demo/mlp.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o demo/mlp $(LDFLAGS)

# pattern rule: any .c -> .o, rebuild if the header changes
%.o: %.c tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

# build and run all tests
test: $(OBJ)
	$(CC) $(CFLAGS) tests/test_tensor.c $(OBJ) -o tests/test_tensor $(LDFLAGS)
	./tests/test_tensor
	$(CC) $(CFLAGS) tests/test_ops.c $(OBJ) -o tests/test_ops $(LDFLAGS)
	./tests/test_ops
	$(CC) $(CFLAGS) tests/test_matmul.c $(OBJ) -o tests/test_matmul $(LDFLAGS)
	./tests/test_matmul
	$(CC) $(CFLAGS) tests/test_views.c $(OBJ) -o tests/test_views $(LDFLAGS)
	./tests/test_views
	$(CC) $(CFLAGS) tests/test_nn.c $(OBJ) -o tests/test_nn $(LDFLAGS)
	./tests/test_nn
	$(CC) $(CFLAGS) tests/test_mismatch.c $(OBJ) -o tests/test_mismatch $(LDFLAGS)
	@echo "== shape mismatch (expect error message + non-zero exit) =="
	@if ./tests/test_mismatch; then echo "[FAIL] did not exit on mismatch"; exit 1; else echo "[PASS] exited non-zero on shape mismatch"; fi
	$(CC) $(CFLAGS) tests/test_matmul_mismatch.c $(OBJ) -o tests/test_matmul_mismatch $(LDFLAGS)
	@echo "== matmul inner-dim mismatch (expect error message + non-zero exit) =="
	@if ./tests/test_matmul_mismatch; then echo "[FAIL] did not exit on inner-dim mismatch"; exit 1; else echo "[PASS] exited non-zero on inner-dim mismatch"; fi
	$(CC) $(CFLAGS) tests/test_reshape_mismatch.c $(OBJ) -o tests/test_reshape_mismatch $(LDFLAGS)
	@echo "== reshape count mismatch (expect error message + non-zero exit) =="
	@if ./tests/test_reshape_mismatch; then echo "[FAIL] did not exit on reshape mismatch"; exit 1; else echo "[PASS] exited non-zero on reshape mismatch"; fi

# same as test, but with sanitizers on to catch memory errors / UB
memcheck: CFLAGS += -fsanitize=address,undefined
memcheck: clean test

# matmul benchmark, built from source with -O2 so the numbers are meaningful
bench: demo/benchmark.c $(SRC)
	$(CC) $(CFLAGS) -O2 $^ -o demo/benchmark $(LDFLAGS)
	./demo/benchmark

clean:
	rm -f $(OBJ) demo/mlp demo/benchmark tests/test_tensor tests/test_ops tests/test_matmul tests/test_views tests/test_nn tests/test_mismatch tests/test_matmul_mismatch tests/test_reshape_mismatch
