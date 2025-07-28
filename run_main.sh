# Multivariate & Lookback = 720
sh scripts/SegRNN/etth1.sh;
sh scripts/SegRNN/etth2.sh;
sh scripts/SegRNN/ettm1.sh;
sh scripts/SegRNN/ettm2.sh;
sh scripts/SegRNN/weather.sh;
sh scripts/SegRNN/solar.sh;
sh scripts/SegRNN/electricity.sh;
sh scripts/SegRNN/traffic.sh;

# Univariate & Lookback = 720
sh scripts/SegRNN/univariate/etth1.sh;
sh scripts/SegRNN/univariate/etth2.sh;
sh scripts/SegRNN/univariate/ettm1.sh;
sh scripts/SegRNN/univariate/ettm2.sh;


# Multivariate & Lookback = 96
sh scripts/SegRNN/Lookback_96/etth1.sh;
sh scripts/SegRNN/Lookback_96/etth2.sh;
sh scripts/SegRNN/Lookback_96/ettm1.sh;
sh scripts/SegRNN/Lookback_96/ettm2.sh;
sh scripts/SegRNN/Lookback_96/weather.sh;
sh scripts/SegRNN/Lookback_96/electricity.sh;
sh scripts/SegRNN/Lookback_96/traffic.sh;