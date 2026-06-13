%% test_llm_connection.m
%  Verify the OpenRouter API key works and probe every model in the panel.
%  Sends one tiny prompt to each model and reports success, latency, token
%  usage, and the raw reply. Use this to confirm the key/credit before running
%  the full multi-model framework.

clear; clc;
here = fileparts(mfilename('fullpath'));
addpath(genpath(here));

cfg = llm_api_config();
fprintf('==============================================================\n');
fprintf('  OPENROUTER CONNECTION TEST  (%d models)\n', numel(cfg.models));
fprintf('==============================================================\n');

sys_p = 'You are a concise assistant. Answer in one short sentence.';
usr_p = 'Reply with exactly: CONNECTION OK';

fprintf('%-34s | %-4s | %-7s | %-12s | reply\n', 'model', 'ok', 'sec', 'tokens(in/out)');
fprintf('%s\n', repmat('-', 1, 90));
for i = 1:numel(cfg.models)
    m = cfg.models{i};
    [txt, info] = call_llm(usr_p, sys_p, cfg.api_key, m);
    reply = regexprep(strtrim(txt), '\s+', ' ');
    if numel(reply) > 30, reply = [reply(1:30) '...']; end
    fprintf('%-34s | %-4d | %-7.2f | %4d/%-7d | %s\n', ...
        m, info.ok, info.elapsed_s, ...
        nz(info.prompt_tokens), nz(info.completion_tokens), reply);
    if ~info.ok && ~isempty(info.error)
        fprintf('    error: %s\n', info.error);
    end
end
fprintf('\nDONE. (If a model fails, check the slug at openrouter.ai/models or your credit.)\n');

function v = nz(x)
    if isnan(x), v = 0; else, v = x; end
end
