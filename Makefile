# EC7207 HPC Group 11 - Pearson Correlation Recommender System
# Makefile for Serial and OpenMP implementations

CC = gcc
CFLAGS = -O2 -Wall
OMPFLAGS = -fopenmp
LFLAGS = -lm

# Targets
SERIAL = serial_rec
OPENMP = openmp_rec

.PHONY: all clean

all: $(SERIAL) $(OPENMP)

$(SERIAL): serial_recommender.c
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)
	@echo "✓ Built: $(SERIAL)"

$(OPENMP): openmp_recommender.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $^ $(LFLAGS)
	@echo "✓ Built: $(OPENMP)"

clean:
	rm -f $(SERIAL) $(OPENMP)
	@echo "✓ Cleaned"

# Test targets
test_serial:
	./$(SERIAL) 500 500

test_openmp:
	OMP_NUM_THREADS=4 ./$(OPENMP) 500 500

test_all: all test_serial test_openmp
