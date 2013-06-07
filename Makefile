APPS = src gpu

all: $(patsubst %,%-all,$(APPS))
clean: $(patsubst %,%-clean,$(APPS))
install: $(patsubst %,%-install,$(APPS))

%-all:
	$(MAKE) -C $(shell echo $@ | cut -f1 -d- )

%-clean:
	$(MAKE) -C $(shell echo $@ | cut -f1 -d- ) clean 

fresh:
	make clean && make

%-install:
	$(MAKE) -C $(shell echo $@ | cut -f1 -d- ) install

