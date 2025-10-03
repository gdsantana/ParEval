#!/usr/bin/env python3
import json
import sys
from collections import Counter

def diagnose_results(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return
    
    print(f"📊 Diagnóstico de: {json_file}")
    print("=" * 80)
    
    total_prompts = len(data)
    print(f"\n✓ Total de prompts: {total_prompts}")
    
    if total_prompts == 0:
        print("❌ Arquivo vazio!")
        return
    
    sample = data[0]
    outputs_per_prompt = len(sample.get('outputs', []))
    print(f"✓ Outputs por prompt: {outputs_per_prompt}")
    
    print(f"\n📝 Campos disponíveis: {list(sample.keys())}")
    
    empty_outputs = 0
    only_braces = 0
    very_short = 0
    normal_outputs = 0
    
    output_lengths = []
    unique_outputs = set()
    
    print("\n🔍 Analisando outputs...")
    for prompt_data in data:
        for output in prompt_data.get('outputs', []):
            output_stripped = output.strip()
            output_lengths.append(len(output_stripped))
            unique_outputs.add(output_stripped)
            
            if len(output_stripped) == 0:
                empty_outputs += 1
            elif output_stripped == '}':
                only_braces += 1
            elif len(output_stripped) < 10:
                very_short += 1
            else:
                normal_outputs += 1
    
    total_outputs = total_prompts * outputs_per_prompt
    
    print(f"\n📈 Estatísticas dos outputs:")
    print(f"  • Total de outputs: {total_outputs}")
    print(f"  • Vazios: {empty_outputs} ({empty_outputs/total_outputs*100:.1f}%)")
    print(f"  • Apenas '}}': {only_braces} ({only_braces/total_outputs*100:.1f}%)")
    print(f"  • Muito curtos (<10 chars): {very_short} ({very_short/total_outputs*100:.1f}%)")
    print(f"  • Normais: {normal_outputs} ({normal_outputs/total_outputs*100:.1f}%)")
    
    if output_lengths:
        avg_length = sum(output_lengths) / len(output_lengths)
        print(f"\n📏 Comprimento médio: {avg_length:.1f} caracteres")
        print(f"  • Mínimo: {min(output_lengths)}")
        print(f"  • Máximo: {max(output_lengths)}")
    
    print(f"\n🔢 Outputs únicos: {len(unique_outputs)}")
    
    if len(unique_outputs) < 10:
        print(f"\n⚠️  PROBLEMA: Apenas {len(unique_outputs)} outputs únicos!")
        print("Outputs únicos encontrados:")
        for i, output in enumerate(list(unique_outputs)[:10], 1):
            display = output[:50] + '...' if len(output) > 50 else output
            print(f"  {i}. {repr(display)}")
    
    print("\n📄 Amostra dos primeiros 3 prompts:")
    for i, prompt_data in enumerate(data[:3], 1):
        print(f"\n--- Prompt {i}: {prompt_data.get('name', 'N/A')} ---")
        print(f"Modelo: {prompt_data.get('parallelism_model', 'N/A')}")
        print(f"Temperatura: {prompt_data.get('temperature', 'N/A')}")
        print(f"do_sample: {prompt_data.get('do_sample', 'N/A')}")
        print(f"prompted: {prompt_data.get('prompted', 'N/A')}")
        
        outputs = prompt_data.get('outputs', [])
        if outputs:
            print(f"\nPrimeiro output ({len(outputs[0])} chars):")
            print(repr(outputs[0][:200]))
            
            if len(outputs) > 1:
                print(f"\nSegundo output ({len(outputs[1])} chars):")
                print(repr(outputs[1][:200]))
    
    print("\n" + "=" * 80)
    
    if only_braces > total_outputs * 0.5:
        print("\n❌ PROBLEMA CRÍTICO DETECTADO!")
        print("   Mais de 50% dos outputs são apenas '}'")
        print("\n💡 Causas prováveis:")
        print("   1. do_sample=False (greedy decoding ignorando temperatura)")
        print("   2. EOS token causando terminação prematura")
        print("   3. Modelo não adequadamente carregado")
        print("\n✅ Soluções:")
        print("   1. Adicione --do_sample ao comando de geração")
        print("   2. Use a configuração corrigida do DeepSeekBaseConfig")
        print("   3. Veja DEEPSEEK_FIX.md para mais detalhes")
    elif normal_outputs > total_outputs * 0.8:
        print("\n✅ Resultados parecem NORMAIS!")
    else:
        print("\n⚠️  Resultados precisam de investigação")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python diagnose_results.py <arquivo.json>")
        print("\nExemplo:")
        print("  python diagnose_results.py results/result-deepseek-coder-1.3b-base-v2.json")
        sys.exit(1)
    
    diagnose_results(sys.argv[1])
