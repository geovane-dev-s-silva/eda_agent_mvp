import os
import subprocess
import sys
import json


def run(cmd):
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()


def check_container(name):
    ps = run(f"docker ps --filter name={name} --format '{{{{.Names}}}}'")
    return name in ps


def main():
    print("🔍 Verificando ambiente Docker EDA...\n")

    backend_ok = check_container("eda_backend")
    frontend_ok = check_container("eda_frontend")

    print(f"🧩 Backend ativo: {'✅' if backend_ok else '❌'}")
    print(f"🧩 Frontend ativo: {'✅' if frontend_ok else '❌'}")

    if not backend_ok or not frontend_ok:
        print("\n⚠️ Um ou mais containers não estão rodando. Execute:")
        print("   docker-compose up -d\n")
        sys.exit(1)

    # Verificar variável no frontend
    env_val = run("docker exec eda_frontend printenv EDA_API_BASE")
    print(f"\n🌐 EDA_API_BASE no frontend: {env_val or '❌ não definida'}")

    # Testar conectividade
    print("\n🔗 Testando conexão do frontend → backend:")
    conn_test = run(
        "docker exec eda_frontend curl -s -o /dev/null -w '%{http_code}' http://backend:8000/api/upload"
    )

    if conn_test == "405":
        print(
            "✅ Backend acessível (retornou código 405 - método não permitido, esperado para GET)."
        )
    elif conn_test == "200":
        print("✅ Backend acessível (200 OK).")
    elif conn_test == "":
        print(
            "❌ Nenhuma resposta. O backend pode não estar rodando ou há erro de rede interna."
        )
    else:
        print(f"⚠️ Resposta inesperada: HTTP {conn_test}")

    # Logs recentes do backend
    print("\n🧾 Últimas 10 linhas de log do backend:")
    logs = run("docker logs --tail 10 eda_backend")
    print(logs or "⚠️ Sem logs disponíveis.")

    print("\n✅ Diagnóstico concluído.\n")


if __name__ == "__main__":
    main()
