# BB 1 DATAB 4a Fase

---

#### Entregável da 4a Fase da Pós Tech FIAP

Participantes:
- Andrea Grassmann
- Victor Almeida
- Eliéser Reis

#### Executar no Windows

- Instale o Python disponível para [download](https://www.python.org/downloads/) para Windows
- Crie um ambiente virtual usando o comando abaixo:
```bash
python -m venv .venv
```
- Ative o ambiente virtual com o comando:
```bash
.venv\Scripts\Activate
```
- Instale as dependências digitando o abaxo:
```bash
pip install -r requirements.txt
```
- Por fim, instale o StreamLit com o comando:
```bash
pip install streamlit
```
- Finalmente, execute o dashboard digitando o seguinte:
```bash
streamlit run
```

#### Executar no Linux ou WSL2

- Instale o Python para sua distribuição Linux _(ou use um gerenciador de pacotes, como **pyenv**, **asdf**, **mise**, etc)_
- Crie um ambiente virtual usando o comando abaixo:
```bash
python -m venv .venv
```
- Ative o ambiente virtual com o comando:
```bash
source .venv/bin/activate
```
- Instale as dependências digitando o abaxo:
```bash
pip install -r requirements.txt
```
- Por fim, instale o StreamLit com o comando:
```bash
pip install streamlit
```
- Finalmente, execute o dashboard digitando o seguinte:
```bash
streamlit run
```

#### Deploy da aplicação
- Este repositório está sendo executado na nuvem da Streamlit em uma conta pessoal grátis conectada a este repositório, no endereço abaixo:
  
  [https://grupo4fase.streamlit.app/](https://grupo4fase.streamlit.app/)

- Alterações que forem comitadas neste repositório são automáticamente executadas na nuvem, por isso execute localmente para testar antes de efetuar o _commit_ das alterações.