from crud import criar_registro, ler_registros, atualizar_registro, deletar_registro

def exibir_menu():
    print("\nMenu:")
    print("1. Criar Registro")
    print("2. Ler Registros")
    print("3. Atualizar Registro")
    print("4. Deletar Registro")
    print("5. Sair")

def main():
    while True:
        exibir_menu()
        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            nome = input("Nome: ")
            email = input("Email: ")
            cidade = input("Cidade: ")
            estado = input("Estado (sigla): ").upper()
            criar_registro(nome, email, cidade, estado)

        elif opcao == '2':
            ler_registros()

        elif opcao == '3':
            registro_id = int(input("ID do registro a atualizar: "))
            nome = input("Novo Nome: ")
            email = input("Novo Email: ")
            cidade = input("Nova Cidade: ")
            estado = input("Novo Estado (sigla): ").upper()
            atualizar_registro(registro_id, nome, email, cidade, estado)

        elif opcao == '4':
            registro_id = int(input("ID do registro a deletar: "))
            deletar_registro(registro_id)

        elif opcao == '5':
            print("Saindo...")
            break

        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
