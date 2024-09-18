.PHONY: register-ipykernel install-with-dev pymc4 pymc5

pymc4:
	ln -sf pyproject-pymc4.toml pyproject.toml
	make install-with-dev

pymc5:
	ln -sf pyproject-pymc5.toml pyproject.toml
	make install-with-dev

install-with-dev:	
	poetry lock
	poetry install --with dev --sync

register-ipykernel:
	poetry run python -m ipykernel install --user --name gppop-neo
