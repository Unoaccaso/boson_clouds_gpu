function [freq_now_s, hnow, fdot_s] = gen_freq_ampl(d, NBh, mu, t0)

%% Constants

G = 6.67e-11;
c = 299792458;
Om0=2*pi/86400; % 1/day
R0=5.5e6; % rotational radius at Livingston (lower latitude)
hbar=1.054571e-34;
onev=1.60217653e-19;
Fint=1e30; % small interaction regime.

duty = 0.70; % detectors duty cycle (approximate)
Tobs=365*86400*duty; % here we should use the exact fraction of non-zero data, taking the smallest among the detectors

%% Setting the BH grid

% BH mass min and max values
Mbh_min=2;
Mbh_mean = 10;
Mbh_max=20;

% BH mass array
triang = makedist("Triangular","a",Mbh_min,"b", Mbh_mean, "c",Mbh_max);
Mbh = triang.random(NBh, 1);

% generate uniform initial BH spins
chi_start=0.2;
chi_stop=0.9;
chi_i = chi_start + (chi_stop-chi_start)*rand(NBh, 1);


%Commenting this for the desired testings
%t0 = zeros(size(Mbh)) + t0
%for i = 1 : length(t0)
%    if rand(1) < .1
%        t0(i) = t0(i) * 1e-2;
%    end
t0s= t0*365*86400;

%% Emitting signals

alpha = G/(c^3*hbar)*2e30*Mbh.*mu*onev; % gravitational 'fine structure constant'
chi_c=4*alpha./(1+4.*alpha.^2); % critical BH spin

tau_inst=27*86400/10*Mbh.*(alpha/0.1).^(-9)./chi_i; % superradiance time scale
tau_gw=6.5e4*365*86400*Mbh./10.*(alpha./0.1).^(-15)./chi_i; % GW emission time scale

freq=483.*(mu/1.0e-12).*(1-0.0056/8*(Mbh./10.).^2.*(mu/1.e-12).^2.); % GW frequency
freq_max = c^3./(2*pi*G*2e30*Mbh).*chi_i./(1+sqrt(1-chi_i.^2)); % maximum allowed GW frequency
fdot = 7e-15*(alpha/0.1).^(17)*(mu/1.0e-12)^2; % spin-up term due to boson annihilations
fdot2 = 1e-10*(10^17/Fint)^4*(alpha/0.1).^(17)*(mu/1.0e-12)^2; % spin-up term due to boson emission
fdot = fdot + fdot2; % total spin-up in the small and intermediate self-interaction regime 
% (as long as Fint < 8.5E16 Gev * (alpha/0.1), when a further term jumps in). The two terms are equal at Fint~1e18 GeV.
freq_now=freq+fdot.*(t0s-tau_inst); % GW frequency at the detector

dec = ceil(freq_now./10)*10;
dfr=Om0*sqrt(2*dec.*R0/c); % search frequency bin
dfdot = dfr./(2*Tobs/duty); % search spin-up half bin 

% conditions to be met in order to have a potentially detectable signal
% (there may be some redundance)
% tau_inst<t0s : superradiance time scale must be shorter than system age
% freq<freq_max : condition for the development of the instability
% 10*tau_inst<tau_gw : we want the instability is fully completed
% chi_i>chi_c : condition for the development of the instability
% freq>20 & freq<610 : GW frequency in the search band
% dfdot > fdot : signal spin-up within half bin
% freq_now < 610: current frequency within search range

ii=find( ...
      tau_inst < 10 * t0s ...
    & freq_now > 20 ...
    & freq_now < 2048 ...
    & 10*tau_inst < tau_gw ...
    & chi_i > chi_c ...
    & dfdot > fdot ...
    & alpha< 0.1 ...
    & tau_gw > 10*t0s ...
    );

if ~isempty(ii) % select subset of parameters satisfying the previous conditions 
    Mbh_s=Mbh(ii);
    alpha_s=alpha(ii);
    chi_i_s=chi_i(ii);
    chi_c_s=chi_c(ii);
    tau_inst_s=tau_inst(ii);
    tau_gw_s=tau_gw(ii);
    freq_now_s=freq_now(ii);
    fdot_s = fdot(ii);

    h0=3.0e-24/10.*Mbh_s.*(alpha_s/0.1).^7.*(chi_i_s-chi_c_s)./0.5; % GW peak amplitude at d=1 kpc
    h0 = h0./d; % actual GW peak amplitude 
    timefactor = (1+(t0s-tau_inst_s)./tau_gw_s); % time-dependent reduction factor
    hnow = h0./timefactor; % actual signal strain amplitude zz + 1;

end
