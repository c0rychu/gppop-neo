.PHONY: register-ipykernel install-with-dev

install-with-dev:	
	poetry install --with dev --sync

register-ipykernel:
	poetry run python -m ipykernel install --user --name gppop-neo
