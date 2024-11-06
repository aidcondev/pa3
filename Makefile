MPICXX = mpic++
CXXFLAGS = -O3

SRC = main.cpp functions.h spgemm.cpp bfs.cpp triangle_counts.cpp
EXEC = pa3

all: $(EXEC)

$(EXEC): $(SRC)
	$(MPICXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(EXEC)
