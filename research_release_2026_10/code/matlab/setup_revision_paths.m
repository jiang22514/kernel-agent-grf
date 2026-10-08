function setup_revision_paths()
% Public, relocatable scientific sources; external GPML is loaded first.
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(here));
gpml=fullfile(root,'third_party','gpml');
assert(isfile(fullfile(gpml,'gp.m')), 'Install GPML 4.2 under third_party/gpml first.');
addpath(genpath(gpml));
addpath(genpath(fullfile(here,'selection')));
addpath(genpath(fullfile(here,'simulation')));
addpath(fullfile(here,'agent'));addpath(fullfile(here,'experiments'));
end
