# Makefile for Recommendation System

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
SRCDIR = .
BINDIR = ./bin
OBJDIR = ./obj

# Source files
SOURCES = main.cpp recommendation.cpp
OBJECTS = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
EXECUTABLE = $(BINDIR)/phase1_serial.exe

# Targets
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "Build complete: $(EXECUTABLE)"

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)
	@echo "Clean complete"

run: $(EXECUTABLE)
	./$(EXECUTABLE)

.PHONY: all clean run
