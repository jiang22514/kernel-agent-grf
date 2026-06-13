function [response_text, info] = call_llm(user_prompt, system_prompt, api_key, model_name)
%CALL_LLM  Call an LLM chat-completion via the OpenRouter API (from MATLAB).
%
%   response_text = call_llm(user_prompt, system_prompt, api_key, model_name)
%   [response_text, info] = call_llm(...)
%
%   OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible endpoint
%   that routes to 300+ models from many providers behind a single key, so the
%   same code can query GPT, Claude, Gemini, DeepSeek, Llama, etc. by changing
%   only the `model_name` slug (e.g. 'openai/gpt-4o-mini',
%   'anthropic/claude-3.5-haiku', 'google/gemini-2.5-flash').
%
%   INPUTS
%     user_prompt    user message text
%     system_prompt  system message text
%     api_key        OpenRouter API key (sk-or-v1-...). If empty, taken from
%                    llm_api_config().
%     model_name     OpenRouter model slug. If empty, default from llm_api_config().
%
%   OUTPUTS
%     response_text  assistant message content ('' on failure)
%     info           struct: .ok, .model, .prompt_tokens, .completion_tokens,
%                    .total_tokens, .elapsed_s, .error
%
%   See also: llm_api_config, test_llm_connection, call_minimax_llm

    cfg = llm_api_config();
    if nargin < 3 || isempty(api_key),    api_key = cfg.api_key;    end
    if nargin < 4 || isempty(model_name), model_name = cfg.models{1}; end

    url = 'https://openrouter.ai/api/v1/chat/completions';

    request_body = struct();
    request_body.model = model_name;
    request_body.messages = { ...
        struct('role', 'system', 'content', system_prompt), ...
        struct('role', 'user',   'content', user_prompt) ...
    };
    request_body.temperature = 0.01;
    request_body.max_tokens  = 4096;
    json_body = jsonencode(request_body);

    opts = weboptions( ...
        'MediaType',     'application/json', ...
        'ContentType',   'json', ...
        'CharacterEncoding', 'UTF-8', ...
        'HeaderFields',  { ...
            'Authorization', ['Bearer ', api_key]; ...
            'HTTP-Referer',  'https://local.matlab'; ...
            'X-Title',       'GP-Kernel-Selection' }, ...
        'Timeout',       90, ...
        'RequestMethod', 'post');

    info = struct('ok', false, 'model', model_name, 'prompt_tokens', NaN, ...
                  'completion_tokens', NaN, 'total_tokens', NaN, ...
                  'elapsed_s', NaN, 'error', '');
    response_text = '';
    n_try = 3;                 % retry transient network/TLS failures
    t0 = tic;
    try
        resp = [];
        for attempt = 1:n_try
            try
                resp = webwrite(url, json_body, opts);
                break
            catch ME
                if attempt == n_try, rethrow(ME); end
                warning('call_llm:retry', 'attempt %d/%d failed (%s); retrying...', ...
                        attempt, n_try, ME.message);
                pause(5 * attempt);
            end
        end
        info.elapsed_s = toc(t0);

        if isfield(resp, 'choices') && ~isempty(resp.choices)
            if iscell(resp.choices), msg = resp.choices{1}; else, msg = resp.choices(1); end
            if isfield(msg, 'message') && isfield(msg.message, 'content')
                response_text = msg.message.content;
                info.ok = true;
            else
                response_text = jsonencode(msg);
            end
        elseif isfield(resp, 'error')
            info.error = stringify(resp.error);
            warning('call_llm:apiError', 'OpenRouter error: %s', info.error);
        else
            response_text = jsonencode(resp);
            warning('call_llm:unexpected', 'Unexpected API response structure.');
        end

        if isfield(resp, 'usage')
            u = resp.usage;
            if isfield(u,'prompt_tokens'),     info.prompt_tokens     = u.prompt_tokens;     end
            if isfield(u,'completion_tokens'), info.completion_tokens = u.completion_tokens; end
            if isfield(u,'total_tokens'),      info.total_tokens      = u.total_tokens;      end
        end
    catch ME
        info.elapsed_s = toc(t0);
        info.error = ME.message;
        warning('call_llm:failed', 'OpenRouter API call failed (%s): %s', ...
                model_name, ME.message);
        response_text = '';
    end
end

% ------------------------------------------------------------------------
function s = stringify(v)
    if ischar(v), s = v;
    elseif isstring(v), s = char(v);
    else
        try, s = jsonencode(v); catch, s = '<unprintable error>'; end
    end
end
