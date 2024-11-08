# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++11

# Source files
SRCS = gpad.cpp acceldualgrad.cpp

# Object files directory
OBJDIR = build

# Object files
OBJS = $(SRCS:%.cpp=$(OBJDIR)/%.o)

# Executable name
TARGET = main

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files to object files
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean