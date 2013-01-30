APPS = src utils gpu

all: $(patsubst %,%-all,$(APPS))
clean: $(patsubst %,%-clean,$(APPS))

%-all:
	$(MAKE) -C $(shell echo $@ | cut -f1 -d- )

%-clean:
	$(MAKE) -C $(shell echo $@ | cut -f1 -d- ) clean 

fresh:
	make clean && make
