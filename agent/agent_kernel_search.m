function out = agent_kernel_search(x, y, opts)
%AGENT_KERNEL_SEARCH  Closed-loop LLM agent for GP kernel/mean selection.
%
%   out = agent_kernel_search(x, y, opts)
%
%   Implements the Perceive -> Plan -> Act -> Reflect loop:
%     Perceive : numeric data profile (describe_dataset) -- no raw data,
%                no ground-truth leakage.
%     Plan     : the LLM proposes a batch of (kernel, mean) candidates from
%                the bounded grammar, in strict JSON.
%     Act      : each new valid candidate is actually fitted
%                (evaluate_expr_aic, exact marginal likelihood) -> AIC/BIC.
%     Reflect  : the full ranked result table (plus rejected-proposal
%                diagnostics) is fed back; the LLM either proposes new
%                candidates or declares convergence with a final pick.
%
%   The loop is stateless on the API side: every round's prompt contains the
%   data profile and the complete evaluated table, which IS the agent state.
%
%   OPTS (all optional except none):
%     .model       OpenRouter slug             (default: llm_api_config 1st)
%     .case_label  dataset label for prompts/logs        (default 'dataset')
%     .max_rounds  max LLM rounds                        (default 4)
%     .n_propose   candidates requested per round        (default 5)
%     .eval_opts   passed to evaluate_expr_aic
%                  (default n_restart=5, n_iter=-100, quiet=true)
%     .log_file    transcript text file ('' = none)
%     .verbose     console progress                      (default true)
%
%   OUT:
%     .best        best candidate found (.name .mean_name .aic .bic .hyp)
%     .final_llm   the LLM's own final recommendation (may differ from .best)
%     .table       struct array of every evaluated candidate
%     .rounds      per-round log (raw reply, parsed proposals, timings)
%     .traj        best-AIC-so-far after each round (regret curve input)
%     .n_evals     number of GP fits actually performed
%     .tokens      cumulative token usage
%     .stopped_by  'llm-stop' | 'max-rounds' | 'stalled'
%
%   See also: describe_dataset, parse_kernel_proposal, evaluate_expr_aic

    if nargin < 3, opts = struct(); end
    cfg = llm_api_config();
    if ~isfield(opts,'model'),      opts.model      = cfg.models{1}; end
    if ~isfield(opts,'case_label'), opts.case_label = 'dataset';     end
    if ~isfield(opts,'max_rounds'), opts.max_rounds = 4;             end
    if ~isfield(opts,'n_propose'),  opts.n_propose  = 5;             end
    if ~isfield(opts,'eval_opts')
        opts.eval_opts = struct('n_restart',5,'n_iter',-100,'quiet',true);
    end
    if ~isfield(opts,'log_file'),   opts.log_file   = '';            end
    if ~isfield(opts,'verbose'),    opts.verbose    = true;          end

    mean_names = {'Zero','Constant','Linear','Quadratic'};
    logf = make_logger(opts.log_file);

    sys_prompt = system_prompt_text();
    profile = describe_dataset(x, y, opts.case_label);
    logf('=== SYSTEM PROMPT ===\n%s\n\n=== DATA PROFILE ===\n%s\n', ...
         sys_prompt, profile);

    % agent state
    T = struct('name',{},'mean_id',{},'mean_name',{},'aic',{},'bic',{}, ...
               'k',{},'ok',{},'round',{},'t_fit',{},'hyp',{});
    seen = containers.Map('KeyType','char','ValueType','logical');
    rounds = struct('reply',{},'props',{},'n_new',{},'t_llm',{},'tokens',{});
    traj = []; tokens = 0; stalled = 0; stopped_by = 'max-rounds';
    final_llm = struct('name','','mean_name','');

    for rd = 1:opts.max_rounds
        % ---------------- Plan (LLM) ----------------
        user_prompt = build_user_prompt(profile, T, rd, opts, mean_names);
        logf('=== ROUND %d : USER PROMPT ===\n%s\n', rd, user_prompt);
        [reply, info] = call_llm(user_prompt, sys_prompt, '', opts.model);
        logf('=== ROUND %d : LLM REPLY (%s, %.1fs, %d tok) ===\n%s\n', ...
             rd, opts.model, info.elapsed_s, nz(info.total_tokens), reply);
        if ~info.ok
            warning('agent_kernel_search:llm', 'LLM call failed: %s', info.error);
            stopped_by = 'llm-error'; break
        end
        tokens = tokens + nz(info.total_tokens);

        [props, raw] = parse_kernel_proposal(reply);

        % ---------------- Act (evaluate new valid candidates) ----------------
        n_new = 0;
        for i = 1:numel(props)
            p = props(i);
            if ~p.valid, continue; end
            key = sprintf('%s|%d', p.name, p.mean_id);
            if isKey(seen, key), continue; end
            seen(key) = true; n_new = n_new + 1;
            t0 = tic;
            r = evaluate_expr_aic(p.name, p.mean_id, x, y, opts.eval_opts);
            tf = toc(t0);
            T(end+1) = struct('name',p.name,'mean_id',p.mean_id, ...
                'mean_name',p.mean_name,'aic',r.aic,'bic',r.bic,'k',r.k, ...
                'ok',r.ok,'round',rd,'t_fit',tf,'hyp',r.hyp); %#ok<AGROW>
            if opts.verbose
                fprintf('  [R%d] %-10s + %-9s  AIC=%9.2f  (%.1fs)\n', ...
                        rd, p.name, p.mean_name, r.aic, tf);
            end
        end
        rounds(end+1) = struct('reply',reply,'props',props,'n_new',n_new, ...
            't_llm',info.elapsed_s,'tokens',nz(info.total_tokens)); %#ok<AGROW>
        traj(end+1) = min([T([T.ok]).aic, inf]); %#ok<AGROW>
        logf('=== ROUND %d : %d new candidates evaluated, best AIC so far %.2f ===\n\n', ...
             rd, n_new, traj(end));

        % ---------------- Reflect / stop ----------------
        % "final" = the agent's current best guess, refreshed every round so a
        % formal recommendation exists regardless of how the loop terminates.
        if isfield(raw,'final') && isstruct(raw.final)
            if isfield(raw.final,'kernel'), final_llm.name = char(raw.final.kernel); end
            if isfield(raw.final,'mean'),   final_llm.mean_name = char(raw.final.mean); end
        end
        if isfield(raw,'stop') && isequal(raw.stop, true)
            stopped_by = 'llm-stop';
            break
        end
        if n_new == 0, stalled = stalled + 1; else, stalled = 0; end
        if stalled >= 2, stopped_by = 'stalled'; break; end
    end

    % ---------------- outputs ----------------
    out = struct();
    out.table = T; out.rounds = rounds; out.traj = traj;
    out.n_evals = numel(T); out.tokens = tokens;
    out.stopped_by = stopped_by; out.final_llm = final_llm;
    okA = [T.ok]; aics = [T.aic]; aics(~okA) = inf;
    [~, ib] = min(aics);
    if ~isempty(T), out.best = T(ib); else, out.best = []; end
    out.model = opts.model; out.case_label = opts.case_label;
    logf('=== DONE: %s | best %s + %s AIC=%.2f | %d evals | %d tokens | %s ===\n', ...
         opts.case_label, T(ib).name, T(ib).mean_name, T(ib).aic, ...
         numel(T), tokens, stopped_by);
end

% ==========================================================================
function s = system_prompt_text()
    s = sprintf([ ...
'You are an expert in Gaussian process (GP) modeling of spatial/geotechnical data.\n' ...
'Goal: find the (kernel, mean) pair with the LOWEST AIC for the described dataset.\n' ...
'You propose candidates; an external system fits each by exact marginal likelihood\n' ...
'(multi-restart) and reports AIC back to you. Fits are expensive: be strategic,\n' ...
'diversify early, then exploit the structure the ranking reveals.\n' ...
'\n' ...
'KERNEL GRAMMAR (strict; anything else is rejected):\n' ...
'- Primitives: SE (squared-exp, smooth), MA1 (Matern-1/2, rough), MA3 (Matern-3/2),\n' ...
'  MA5 (Matern-5/2), RQ (rational quadratic, multi-scale), LIN (linear, nonstationary),\n' ...
'  PER (periodic). All ARD (per-dimension lengthscales) where applicable.\n' ...
'- An expression is ONE primitive, or TWO DISTINCT primitives joined by + or *.\n' ...
'- Products of two stationary primitives (SE,MA1,MA3,MA5,RQ) are FORBIDDEN\n' ...
'  (redundant), e.g. SE*MA3 invalid. Valid products pair a stationary with LIN or\n' ...
'  PER (e.g. MA1*LIN = locally rough with linearly growing variance), or LIN*PER.\n' ...
'- Examples of valid: "SE", "RQ", "SE+PER", "MA3+LIN", "MA1*LIN", "LIN*PER".\n' ...
'  Invalid: "SE*RQ", "SE+SE", "SE+MA1+LIN", "SE*(LIN+PER)".\n' ...
'MEAN FUNCTIONS: "Zero" | "Constant" | "Linear" | "Quadratic" (polynomial trends\n' ...
'include an intercept). Kernel and mean interact: e.g. a LIN kernel or a Linear\n' ...
'mean can both capture trend; AIC penalizes redundant parameters.\n' ...
'\n' ...
'RESPONSE FORMAT -- reply with ONE JSON object and NOTHING else:\n' ...
'{"analysis":"<=60 words on what the evidence suggests",\n' ...
' "proposals":[{"kernel":"MA1*LIN","mean":"Constant","reason":"<=20 words"}, ...],\n' ...
' "stop":false,\n' ...
' "final":{"kernel":"...","mean":"..."}}\n' ...
'Rules: never re-propose a candidate already in the results table. "final" is\n' ...
'REQUIRED in EVERY round and holds your current best guess given the evidence so\n' ...
'far. Set "stop":true only when further search is unlikely to lower AIC.\n']);
end

% ==========================================================================
function up = build_user_prompt(profile, T, rd, opts, mean_names) %#ok<INUSD>
    if rd == 1
        up = sprintf(['%s\n\nNo candidates evaluated yet. Propose %d diverse ' ...
            '(kernel, mean) candidates covering different structural hypotheses ' ...
            '(trend vs stationary, smooth vs rough, with/without periodicity).\n' ...
            'This is round 1 of at most %d.'], ...
            profile, opts.n_propose, opts.max_rounds);
        return
    end
    % ranked results table
    okA = [T.ok]; aics = [T.aic]; aics(~okA) = inf;
    [~, si] = sort(aics);
    rows = cell(numel(T),1);
    best = aics(si(1));
    for r = 1:numel(T)
        t = T(si(r));
        if t.ok
            rows{r} = sprintf('%2d. %-10s + %-9s  AIC=%9.2f  dAIC=%7.2f  (k=%d, round %d)', ...
                              r, t.name, t.mean_name, t.aic, t.aic-best, t.k, t.round);
        else
            rows{r} = sprintf('%2d. %-10s + %-9s  FIT FAILED', r, t.name, t.mean_name);
        end
    end
    up = sprintf(['%s\n\nRESULTS AFTER ROUND %d (sorted by AIC, lower is better):\n%s\n\n' ...
        'This is round %d of at most %d. Reflect on the ranking: which structural ' ...
        'ingredients (roughness class, LIN/PER components, mean order) does the ' ...
        'evidence favor? Then either propose up to %d NEW candidates (not in the ' ...
        'table) that could lower AIC further, or set "stop":true with your "final" pick.'], ...
        profile, rd-1, strjoin(rows, newline), rd, opts.max_rounds, opts.n_propose);
end

% ==========================================================================
function logf = make_logger(fname)
    if isempty(fname)
        logf = @(varargin) [];
    else
        fid = fopen(fname, 'a');
        logf = @(varargin) fprintf(fid, varargin{:});
    end
end

function v = nz(v), if isnan(v), v = 0; end, end
