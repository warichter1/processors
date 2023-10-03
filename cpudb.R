# read in CSV tables
processor  <- read.csv("./cpudb/processor.csv")
specint2k6 <- read.csv("./cpudb/spec_int2006.csv")
specint2k0 <- read.csv("./cpudb/spec_int2000.csv")
specint95 <- read.csv("./cpudb/spec_int1995.csv")
specint92 <- read.csv("./cpudb/spec_int1992.csv")

# merge spec scores
all <- merge(processor, specint2k6, by="processor_id",
             suffixes=c(".proc", ".spec_int2k6"), all=TRUE)
all <- merge(all, specint2k0, by="processor_id",
             suffixes=c(".spec_int2k6", ".spec_int2k0"), all=TRUE)
all <- merge(all, specint95, by="processor_id",
             suffixes=c(".spec_int2k0", ".spec_int95"), all=TRUE)
all <- merge(all, specint92, by="processor_id",
             suffixes=c(".spec_int95", ".spec_int92"), all=TRUE)

# fix missing date entries 
all[all[["date"]]=="","date"] <- NA
dates <- as.POSIXct(all[["date"]])

# account for potential turbo-boost clock
noturbo <- is.na(all[["max_clock"]])
all[noturbo,"max_clock"] <- all[noturbo, "clock"]

# determine scaling factors for spec92->spec95,
# spec95->spec2k0, and spec2k0->spec2k6
spec92to95 <- mean(all[["basemean.spec_int95"]]/all[["basemean.spec_int92"]], na.rm=TRUE)
spec95to2k0 <- mean(all[["basemean.spec_int2k0"]]/all[["basemean.spec_int95"]], na.rm=TRUE)
spec2k0to2k6 <- mean(all[["basemean.spec_int2k6"]]/all[["basemean.spec_int2k0"]], na.rm=TRUE)

no95 <- is.na(all[["basemean.spec_int95"]])
no2k0 <- is.na(all[["basemean.spec_int2k0"]])
no2k6 <- is.na(all[["basemean.spec_int2k6"]])
all[no95, "basemean.spec_int95"] <- spec92to95 * all[no95, "basemean.spec_int92"]
all[no2k0,"basemean.spec_int2k0"] <- spec95to2k0 * all[no2k0, "basemean.spec_int95"]
all[no2k6, "basemean.spec_int2k6"] <- spec2k0to2k6 * all[no2k6, "basemean.spec_int2k0"]

# performance
all[["perfnorm"]] <- all[["basemean.spec_int2k6"]]/all[["tdp"]]

# find the scaling factors
scaleclk   <- min(all[["max_clock"]], na.rm=TRUE)
scaletrans <- min(all[["transistors"]], na.rm=TRUE)
scaletdp   <- min(all[["tdp"]], na.rm=TRUE)
scaleperf  <- min(all[["basemean.spec_int2k6"]], na.rm=TRUE)
scaleperfnorm  <- min(all[["perfnorm"]], na.rm=TRUE)

# make the plot
plot(dates, all[["transistors"]]/scaletrans, log="y", col=1, bg=1, pch=22,
     cex=0.7, ylab="Relative scaling", main="Processor scaling trends")
points(dates, all[["max_clock"]]/scaleclk, col=2, bg=2, pch=20, cex=0.7)
points(dates, all[["tdp"]]/scaletdp, col=3, bg=3, pch=24, cex=0.7)
points(dates, all[["basemean.spec_int2k6"]]/scaleperf, col=4, bg=4, pch=20, cex=0.7)
points(dates, all[["perfnorm"]]/scaleperfnorm, col=5, bg=5, pch=20, cex=0.7)
legend(as.POSIXct("1975-01-01"), y=1e6,
       c("Transistors", "Clock", "Power", "Performance", "Performance/W"),
       col=c(1,2,3,4,5), pt.bg=c(1,2,3,4,5), pch=c(22,20,24,20,20), pt.cex=0.7)
