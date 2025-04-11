from flask import Flask, jsonify, request, make_response
import jwt
import datetime
from functools import wraps

app = Flask(__name__)

# Chave secreta para criptografia JWT
app.config['SECRET_KEY'] = 'teste'

# Dados simulados
dados = [
    {"id": 1, "nome": "Produto 1", "preco": 50},
    {"id": 2, "nome": "Produto 2", "preco": 150}
]

teste_token = [{ "username": "admin", "password": "senha", "token": "teste" }]

# Função para verificar o token em rotas protegidas
def token_requerido(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"message": "Token é necessário"}), 403
        
        try:
            # Decodificar o token JWT
            jwt.decode(token.split(" ")[1], app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({"message": "Token inválido ou expirado"}), 403
        
        return f(*args, **kwargs)
    return decorator

API_KEY = 'teste'

@app.before_request
def authenticate():
    key = request.headers.get('x-api-key')
    if key != API_KEY:
        return jsonify({"message": "Unauthorized"}), 403


# Rota de login para gerar o token JWT
@app.route('/login', methods=['POST'])
def login():
    # Apenas uma simulação, você pode incluir uma validação com um banco de dados
    if request.json.get("username") == "admin" and request.json.get("password") == "senha":
        token = jwt.encode({
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({"token": token}), 200
    
    return make_response("Credenciais inválidas", 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})

# Rota para retornar todos os produtos (GET) - sem autenticação
@app.route('/produtos', methods=['GET'])
def get_produtos():
    return jsonify(dados)

# Rota para buscar um produto específico por ID (GET) - sem autenticação
@app.route('/produtos/<int:id>', methods=['GET'])
def get_produto(id):
    produto = next((item for item in dados if item['id'] == id), None)
    if produto:
        return jsonify(produto)
    else:
        return jsonify({"erro": "Produto não encontrado"}), 404

# Rota para criar um novo produto (POST) - requer token
@app.route('/produtos', methods=['POST'])
@token_requerido
def add_produto():
    novo_produto = request.json
    dados.append(novo_produto)
    return jsonify(novo_produto), 201

# Rota para atualizar um produto existente (PUT) - requer token
@app.route('/produtos/<int:id>', methods=['PUT'])
@token_requerido
def update_produto(id):
    produto = next((item for item in dados if item['id'] == id), None)
    if produto:
        produto.update(request.json)
        return jsonify(produto)
    else:
        return jsonify({"erro": "Produto não encontrado"}), 404

# Rota para deletar um produto (DELETE) - requer token
@app.route('/produtos/<int:id>', methods=['DELETE'])
@token_requerido
def delete_produto(id):
    global dados
    dados = [item for item in dados if item['id'] != id]
    return jsonify({"message": "Produto deletado"}), 200

if __name__ == '__main__':
    app.run(debug=False)
