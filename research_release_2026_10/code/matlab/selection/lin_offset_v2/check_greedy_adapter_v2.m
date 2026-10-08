function report=check_greedy_adapter_v2()
% Regression of search order against the existing five-table v3 baseline.
here=fileparts(mfilename('fullpath'));root=fileparts(fileparts(fileparts(here)));
addpath(fullfile(root,'kernel-agent-grf'));setup_revision_paths;addpath(here);maxNumCompThreads(1);
cases={'EX1','EX2','EX3','EX4','Borehole'};caps=[20,40,80,160,500];checks=cell(1,5);
for ci=1:5
    sf=fullfile(root,'output','revision','07_整改_B7','scores_v3',['score_' cases{ci} '_scoring_v3.mat']);
    T=load(sf,'recs');n=zeros(1,5);
    for bi=1:5
        o=struct('budget',caps(bi),'width',2,'table_only',true,'score_file',sf);
        a=greedy_beam_v3(cases{ci},o);b=greedy_beam_lin_v2(T.recs,o);
        assert(isequal(a.seq,b.seq)&&isequal(a.aic,b.aic)&&isequal(a.traj,b.traj)&&isequal(a.layer,b.layer));
        assert(isequal(a.beam_hist,b.beam_hist)&&a.cache_stats.misses==0&&b.cache_stats.misses==0);
        n(bi)=b.evals_used;
    end
    checks{ci}=struct('case',cases{ci},'caps',caps,'accesses',n,'all_trajectories_identical',true);
end
messages=checkcode(fullfile(here,'greedy_beam_lin_v2.m'),'-id');
driver_messages=checkcode(fullfile(here,'run_cheap_baselines_lin_v2.m'),'-id');
report=struct('pass',true,'case_checks',{checks},'code_messages',messages,'driver_messages',driver_messages,'new_fits',0,'API_calls',0);
out=fullfile(root,'output','revision','review_fixes_2026-10-06','lin_v2','baselines');if ~isfolder(out),mkdir(out);end
f=fopen(fullfile(out,'adapter_regression.json'),'w');assert(f>=0);c=onCleanup(@()fclose(f));
fwrite(f,unicode2native(jsonencode(report,PrettyPrint=true),'UTF-8'),'uint8');
fprintf('GREEDY_V2_ADAPTER_PASS cases=5 caps=5 trajectories=25 new_fits=0\n');
end
