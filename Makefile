UV_CC=uv pip compile pyproject.toml --prerelease allow
UV_OPTS=--extra-index-url=https://$(FURY_AUTH):@pypi.fury.io/sarus

all: uv-clean lock

uv-clean:
	uv cache clean

lock:
	$(UV_CC) $(UV_OPTS) -o requirements.txt

.venv:
	uv venv --python 3.10

venv: .venv
	uv pip sync requirements.txt $(UV_OPTS) && \
	bash -c 'source $(CURDIR)/.venv/bin/activate' && \
	uv pip install --upgrade pip #uv is not installing pip by default
