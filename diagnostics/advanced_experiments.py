import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from scipy.stats import entropy
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import inspect

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentD_RandomGates:
    """
    Experiment D: compare adaptive, fixed_all_on, fixed_random.

    If fixed_all_on ≈ adaptive, learned gates add little.
    If fixed_random ≈ adaptive, learned policy is near-random.
    """
    
    @staticmethod
    def generate_random_gates_matching_length(adaptive_gates, target_effective_length=None):
        """
        Random gates with the same effective length as the adaptive schedule.

        Args:
            adaptive_gates: Current gate logits.
            target_effective_length: Target count of gates > 0.5 after sigmoid (None = match adaptive).
        """
        if target_effective_length is None:
            gates_sigmoid = torch.sigmoid(adaptive_gates)
            target_effective_length = int((gates_sigmoid > 0.5).sum().item())
        
        n_gates = len(adaptive_gates)
        
        # Random permutation: first target_effective_length entries high, rest low
        random_perm = torch.randperm(n_gates)
        random_gates = torch.ones_like(adaptive_gates) * (-2.0)  # sigmoid ≈ 0.12
        random_gates[random_perm[:target_effective_length]] = 2.0  # sigmoid ≈ 0.88
        
        random_gates += torch.randn_like(random_gates) * 0.5
        
        return random_gates

    @staticmethod
    def test_random_vs_adaptive(model, val_loader, device, n_random_trials=5):
        """Compare adaptive, all-on, and several random gate draws."""
        model.eval()
        prompt_learner = model.prompt_learner

        # Snapshot adaptive gates
        if not (hasattr(prompt_learner, 'length_gates') and isinstance(prompt_learner.length_gates, nn.Parameter)):
            return {'error': 'No adaptive gates found'}

        adaptive_length_gates = prompt_learner.length_gates.data.clone()
        adaptive_length_device = prompt_learner.length_gates.device
        adaptive_length_grad = prompt_learner.length_gates.requires_grad

        adaptive_depth_weights = None
        adaptive_depth_device = None
        adaptive_depth_grad = None
        if hasattr(prompt_learner, 'depth_weights') and isinstance(prompt_learner.depth_weights, nn.Parameter):
            adaptive_depth_weights = prompt_learner.depth_weights.data.clone()
            adaptive_depth_device = prompt_learner.depth_weights.device
            adaptive_depth_grad = prompt_learner.depth_weights.requires_grad

        results = {}

        try:
            print("    Evaluating adaptive...")
            model.train()
            acc_adaptive = ExperimentD_RandomGates._evaluate_accuracy(model, val_loader, device)
            model.eval()
            results['adaptive'] = acc_adaptive

            temperature = getattr(prompt_learner, 'temperature', 1.0)
            adaptive_sigmoid = torch.sigmoid(adaptive_length_gates / temperature)
            effective_length = int((adaptive_sigmoid > 0.5).sum().item())
            results['adaptive_effective_length'] = effective_length

            print("    Evaluating fixed_all_on...")
            prompt_learner.length_gates.data = torch.ones_like(adaptive_length_gates) * 2.0
            if adaptive_depth_weights is not None:
                prompt_learner.depth_weights.data = torch.ones_like(adaptive_depth_weights) * 2.0
            model.train()
            acc_all_on = ExperimentD_RandomGates._evaluate_accuracy(model, val_loader, device)
            model.eval()
            results['fixed_all_on'] = acc_all_on

            print(f"    Evaluating {n_random_trials} fixed_random trials (effective_length={effective_length})...")
            random_accs = []
            for trial in range(n_random_trials):
                random_length_gates = ExperimentD_RandomGates.generate_random_gates_matching_length(
                    adaptive_length_gates, effective_length
                )
                prompt_learner.length_gates.data = random_length_gates.to(adaptive_length_device)

                if adaptive_depth_weights is not None:
                    random_depth_weights = ExperimentD_RandomGates.generate_random_gates_matching_length(
                        adaptive_depth_weights, effective_length
                    )
                    prompt_learner.depth_weights.data = random_depth_weights.to(adaptive_depth_device)

                model.train()
                acc_random = ExperimentD_RandomGates._evaluate_accuracy(model, val_loader, device)
                model.eval()
                random_accs.append(acc_random)
                print(f"      Trial {trial + 1}: {acc_random:.2f}%")

            results['fixed_random_trials'] = random_accs
            results['fixed_random_mean'] = np.mean(random_accs)
            results['fixed_random_std'] = np.std(random_accs)
            results['fixed_random_min'] = np.min(random_accs)
            results['fixed_random_max'] = np.max(random_accs)

            results['analysis'] = {
                'adaptive_vs_all_on': acc_adaptive - acc_all_on,
                'adaptive_vs_random_mean': acc_adaptive - results['fixed_random_mean'],
                'adaptive_vs_random_best': acc_adaptive - results['fixed_random_max'],
                'random_covers_adaptive': results['fixed_random_min'] <= acc_adaptive <= results['fixed_random_max'],
                'conclusion': ExperimentD_RandomGates._draw_conclusion(acc_adaptive, acc_all_on,
                                                                       results['fixed_random_mean'],
                                                                       results['fixed_random_max'])
            }

        finally:
            print("    Restoring adaptive gates...")
            prompt_learner.length_gates.data = adaptive_length_gates.to(adaptive_length_device)
            prompt_learner.length_gates.requires_grad = adaptive_length_grad

            if adaptive_depth_weights is not None:
                prompt_learner.depth_weights.data = adaptive_depth_weights.to(adaptive_depth_device)
                prompt_learner.depth_weights.requires_grad = adaptive_depth_grad

            if not torch.allclose(prompt_learner.length_gates.data, adaptive_length_gates.to(adaptive_length_device),
                                  atol=1e-6):
                print("    Warning: length_gates not properly restored, forcing restore...")
                prompt_learner.length_gates = nn.Parameter(adaptive_length_gates.to(adaptive_length_device))
                prompt_learner.length_gates.requires_grad = adaptive_length_grad

        return results
    
    @staticmethod
    def _draw_conclusion(acc_adaptive, acc_all_on, acc_random_mean, acc_random_max):
        """Build human-readable conclusion strings."""
        conclusions = []
        
        if abs(acc_adaptive - acc_all_on) < 1.0:
            conclusions.append("PASS: adaptive ~ all_on — gates add little; use full-on")
        elif acc_all_on > acc_adaptive + 1.0:
            conclusions.append("PASS: all_on > adaptive — learned gates hurt; prefer full-on")
        
        if abs(acc_adaptive - acc_random_mean) < 1.0:
            conclusions.append("PASS: adaptive ~ random mean — policy near random")
        elif acc_random_max > acc_adaptive:
            conclusions.append("PASS: random_best > adaptive — random search beats learned gates")
        
        if not conclusions:
            conclusions.append("NOT SHOWN: adaptive may be better than baselines")
        
        return conclusions
    
    @staticmethod
    def _evaluate_accuracy(model, val_loader, device, max_batches=50):
        """Top-1 accuracy on val_loader (subset)."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                images = batch['img'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(images)
                _, predicted = logits.max(1)
                
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return 100.0 * correct / total if total > 0 else 0.0


class ExperimentE_SimplifiedGate:
    """
    Experiment E: simplify gates from per-token to per-layer.

    Tests whether per-token gating is unnecessarily redundant.
    """
    
    @staticmethod
    def create_per_layer_gate_model(model):
        """
        Build a per-layer gate variant: collapse length_gates toward a scalar prototype.
        depth_weights unchanged.
        """
        prompt_learner = model.prompt_learner
        
        original_length_gates = None
        if hasattr(prompt_learner, 'length_gates') and isinstance(prompt_learner.length_gates, nn.Parameter):
            original_length_gates = prompt_learner.length_gates.clone()
            
            avg_gate_value = original_length_gates.mean().item()
            
            prompt_learner.length_gates = nn.Parameter(torch.tensor([avg_gate_value]))
            prompt_learner.per_layer_mode = True
        
        return original_length_gates
    
    @staticmethod
    def restore_per_token_gates(model, original_gates):
        """Restore full per-token length_gates."""
        prompt_learner = model.prompt_learner
        if original_gates is not None:
            prompt_learner.length_gates = nn.Parameter(original_gates)
            prompt_learner.per_layer_mode = False

    @staticmethod
    def test_simplified_gate(model, val_loader, device):
        """
        Compare per-token vs uniform per-layer (same value per position).

        Returns:
            dict with accuracy and pattern statistics.
        """
        model.eval()
        results = {}

        print("    Evaluating per-token gate...")
        model.train()
        acc_per_token = ExperimentE_SimplifiedGate._evaluate_accuracy(model, val_loader, device)
        model.eval()
        results['per_token_acc'] = acc_per_token

        prompt_learner = model.prompt_learner
        if hasattr(prompt_learner, 'length_gates') and isinstance(prompt_learner.length_gates, nn.Parameter):
            temperature = getattr(prompt_learner, 'temperature', 1.0)
            gates_sigmoid = torch.sigmoid(prompt_learner.length_gates / temperature).detach().cpu().numpy()
            results['per_token_pattern'] = {
                'values': gates_sigmoid.tolist(),
                'mean': float(gates_sigmoid.mean()),
                'std': float(gates_sigmoid.std()),
                'entropy': float(entropy(gates_sigmoid + 1e-10)),
                'monotonicity': float(np.corrcoef(range(len(gates_sigmoid)), gates_sigmoid)[0, 1]) if len(
                    gates_sigmoid) > 1 else 0.0,
                'has_clear_pattern': gates_sigmoid.std() > 0.15
            }

        original_gates = None
        original_gates_device = None
        original_gates_requires_grad = None

        if hasattr(prompt_learner, 'length_gates') and isinstance(prompt_learner.length_gates, nn.Parameter):
            original_gates = prompt_learner.length_gates.data.clone()
            original_gates_device = prompt_learner.length_gates.device
            original_gates_requires_grad = prompt_learner.length_gates.requires_grad

            avg_gate_value = original_gates.mean().item()

            print("    Switching to per-layer gate...")
            per_layer_gates = torch.full_like(original_gates, avg_gate_value)
            prompt_learner.length_gates.data = per_layer_gates

            print("    Evaluating per-layer gate (mean value)...")
            model.train()
            acc_per_layer_fixed = ExperimentE_SimplifiedGate._evaluate_accuracy(model, val_loader, device)
            model.eval()
            results['per_layer_fixed_acc'] = acc_per_layer_fixed

            print("    Restoring per-token gate...")
            prompt_learner.length_gates.data = original_gates.to(original_gates_device)
            prompt_learner.length_gates.requires_grad = original_gates_requires_grad
        else:
            print("    Warning: No length_gates found, skipping per-layer test")
            results['per_layer_fixed_acc'] = acc_per_token

        if original_gates is not None:
            current_gates = prompt_learner.length_gates.data
            if not torch.allclose(current_gates, original_gates.to(current_gates.device), atol=1e-6):
                print("    Warning: Gates not properly restored!")
                prompt_learner.length_gates = nn.Parameter(original_gates.to(original_gates_device))
                prompt_learner.length_gates.requires_grad = original_gates_requires_grad

        results['analysis'] = {
            'performance_gap': acc_per_token - results['per_layer_fixed_acc'],
            'per_layer_sufficient': abs(acc_per_token - results['per_layer_fixed_acc']) < 1.0,
            'token_level_redundant': results['per_token_pattern'][
                                         'std'] < 0.1 if 'per_token_pattern' in results else False,
            'conclusion': ExperimentE_SimplifiedGate._draw_conclusion(
                acc_per_token,
                results['per_layer_fixed_acc'],
                results.get('per_token_pattern', {})
            )
        }

        return results
    
    @staticmethod
    def _draw_conclusion(acc_per_token, acc_per_layer, pattern_info):
        """English conclusion lines for Experiment E."""
        conclusions = []
        
        gap = acc_per_token - acc_per_layer
        
        if abs(gap) < 1.0:
            conclusions.append("PASS: per-layer enough — per-token may be overly complex")
            conclusions.append(f"  Small accuracy gap ({gap:+.2f}%)")
        
        if pattern_info['std'] < 0.1:
            conclusions.append("PASS: per-token gates nearly uniform — weak token-level structure")
            conclusions.append(f"  Std only {pattern_info['std']:.3f}")
        
        if pattern_info['entropy'] > 2.0:
            conclusions.append("PASS: per-token gates near-uniform — low selectivity")
        
        if not conclusions:
            conclusions.append("NOT SHOWN: per-token pattern looks informative")
        
        return conclusions
    
    @staticmethod
    def _evaluate_accuracy(model, val_loader, device, max_batches=50):
        """Top-1 accuracy on val_loader (subset)."""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                images = batch['img'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(images)
                _, predicted = logits.max(1)
                
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return 100.0 * correct / total if total > 0 else 0.0


class ExperimentF_EmbeddingStability:
    """
    Experiment F: embedding drift and alignment vs CLIP prior.

    Track drift and clustering quality across training.
    """

    @staticmethod
    def extract_embeddings(model, data_loader, device, max_batches=50):
        """
        Extract image and text embeddings (works across several trainer layouts).

        Returns:
            dict with image_embeddings, text_embeddings, labels arrays.
        """
        model.eval()

        image_embeddings = []
        text_embeddings = []
        labels_list = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break

                images = batch['img'].to(device)
                labels = batch['label'].to(device)

                try:
                    if hasattr(model, '__call__'):
                        logits = model(images)

                        if hasattr(model, 'prompt_learner'):
                            prompt_outputs = model.prompt_learner()

                            if len(prompt_outputs) == 6:
                                prompts, shared_ctx, deep_text, deep_vision, depth_probs, length_gates = prompt_outputs
                            elif len(prompt_outputs) == 4:
                                prompts, shared_ctx, deep_text, deep_vision = prompt_outputs
                                depth_probs = None
                                length_gates = None
                            else:
                                raise ValueError(
                                    f"Unexpected number of outputs from prompt_learner: {len(prompt_outputs)}")

                            # Image encoding
                            if hasattr(model, 'image_encoder'):
                                sig = inspect.signature(model.image_encoder.forward)
                                params = list(sig.parameters.keys())

                                if 'depth_probs' in params:
                                    img_emb = model.image_encoder(
                                        images.type(model.dtype),
                                        shared_ctx,
                                        deep_vision,
                                        depth_probs
                                    )
                                else:
                                    img_emb = model.image_encoder(
                                        images.type(model.dtype),
                                        shared_ctx,
                                        deep_vision
                                    )
                            else:
                                img_emb = model.module.image_encoder(images.type(model.dtype))

                            tokenized_prompts = model.prompt_learner.tokenized_prompts.to(device)

                            if hasattr(model, 'text_encoder'):
                                sig = inspect.signature(model.text_encoder.forward)
                                params = list(sig.parameters.keys())

                                if 'depth_weights' in params:
                                    txt_emb = model.text_encoder(
                                        prompts, tokenized_prompts, deep_text, depth_weights=depth_probs
                                    )
                                elif len(params) > 3:
                                    txt_emb = model.text_encoder(
                                        prompts, tokenized_prompts, deep_text, depth_probs
                                    )
                                else:
                                    txt_emb = model.text_encoder(prompts, tokenized_prompts, deep_text)
                            else:
                                txt_emb = model.module.text_encoder(prompts, tokenized_prompts, deep_text, depth_probs)
                        else:
                            img_emb = model.encode_image(images)
                            txt_emb = model.encode_text(labels)

                    img_emb = F.normalize(img_emb, dim=-1)
                    txt_emb = F.normalize(txt_emb, dim=-1)

                    txt_emb_selected = txt_emb[labels]

                    image_embeddings.append(img_emb.cpu())
                    text_embeddings.append(txt_emb_selected.cpu())
                    labels_list.append(labels.cpu())

                except Exception as e:
                    print(f"Warning: Batch {batch_idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        if not image_embeddings:
            return None

        return {
            'image_embeddings': torch.cat(image_embeddings, dim=0).numpy(),
            'text_embeddings': torch.cat(text_embeddings, dim=0).numpy(),
            'labels': torch.cat(labels_list, dim=0).numpy()
        }
    
    @staticmethod
    def compute_embedding_metrics(embeddings):
        """Alignment, clustering, and class-separation metrics."""
        img_emb = embeddings['image_embeddings']
        txt_emb = embeddings['text_embeddings']
        labels = embeddings['labels']
        
        metrics = {}
        
        cosine_sim = np.sum(img_emb * txt_emb, axis=1)
        metrics['alignment'] = {
            'mean': float(cosine_sim.mean()),
            'std': float(cosine_sim.std()),
            'min': float(cosine_sim.min()),
            'max': float(cosine_sim.max())
        }
        
        if len(np.unique(labels)) > 1:
            try:
                img_silhouette = silhouette_score(img_emb, labels, metric='cosine')
                txt_silhouette = silhouette_score(txt_emb, labels, metric='cosine')
                metrics['clustering'] = {
                    'image_silhouette': float(img_silhouette),
                    'text_silhouette': float(txt_silhouette),
                    'avg_silhouette': float((img_silhouette + txt_silhouette) / 2)
                }
            except Exception as e:
                print(f"Warning: Silhouette score computation failed: {e}")
                metrics['clustering'] = {'error': str(e)}
        
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            img_centers = np.array([img_emb[labels == l].mean(axis=0) for l in unique_labels])
            txt_centers = np.array([txt_emb[labels == l].mean(axis=0) for l in unique_labels])
            
            img_center_dists = []
            txt_center_dists = []
            for i in range(len(unique_labels)):
                for j in range(i+1, len(unique_labels)):
                    img_center_dists.append(1 - np.dot(img_centers[i], img_centers[j]))
                    txt_center_dists.append(1 - np.dot(txt_centers[i], txt_centers[j]))
            
            metrics['separation'] = {
                'image_center_dist': float(np.mean(img_center_dists)) if img_center_dists else 0.0,
                'text_center_dist': float(np.mean(txt_center_dists)) if txt_center_dists else 0.0
            }
        
        return metrics
    
    @staticmethod
    def compute_embedding_drift(embeddings_t0, embeddings_t):
        """Drift between two embedding snapshots (1 - cosine similarity per sample)."""
        img_t0 = embeddings_t0['image_embeddings']
        img_t = embeddings_t['image_embeddings']
        txt_t0 = embeddings_t0['text_embeddings']
        txt_t = embeddings_t['text_embeddings']
        
        img_cosine_change = 1 - np.sum(img_t0 * img_t, axis=1)
        txt_cosine_change = 1 - np.sum(txt_t0 * txt_t, axis=1)
        
        drift_metrics = {
            'image_drift': {
                'mean': float(img_cosine_change.mean()),
                'std': float(img_cosine_change.std()),
                'max': float(img_cosine_change.max())
            },
            'text_drift': {
                'mean': float(txt_cosine_change.mean()),
                'std': float(txt_cosine_change.std()),
                'max': float(txt_cosine_change.max())
            },
            'avg_drift': float((img_cosine_change.mean() + txt_cosine_change.mean()) / 2)
        }
        
        return drift_metrics
    
    @staticmethod
    def visualize_embedding_alignment(embeddings, save_path):
        """Save a heatmap of image–text cosine similarities (subsampled)."""
        img_emb = embeddings['image_embeddings']
        txt_emb = embeddings['text_embeddings']
        labels = embeddings['labels']
        
        n_samples = min(100, len(labels))
        indices = np.random.choice(len(labels), n_samples, replace=False)
        
        img_sample = img_emb[indices]
        txt_sample = txt_emb[indices]
        labels_sample = labels[indices]
        
        # Compute similarity matrix
        sim_matrix = img_sample @ txt_sample.T
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                   xticklabels=labels_sample, yticklabels=labels_sample,
                   cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Image-Text Embedding Alignment')
        plt.xlabel('Text Embeddings')
        plt.ylabel('Image Embeddings')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


class AdvancedExperimentRunner:
    """Phase-2 advanced diagnostics (D/E/F)."""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.results = {}
    
    def run_all_experiments(self):
        """Run phase-2 experiments D, E, F."""
        print("="*80)
        print("Running phase-2 diagnostics...")
        print("="*80)
        
        print("\n[Exp D] Fixed vs random gates")
        print("-"*80)
        try:
            self.results['D_random_gates'] = ExperimentD_RandomGates.test_random_vs_adaptive(
                self.model, self.val_loader, self.device, n_random_trials=5
            )
            self._print_experiment_d_summary()
        except Exception as e:
            print(f"FAILED Exp D: {e}")
            import traceback
            traceback.print_exc()
            self.results['D_random_gates'] = {'error': str(e)}
        
        print("\n[Exp E] Per-layer vs per-token gate")
        print("-"*80)
        try:
            self.results['E_simplified_gate'] = ExperimentE_SimplifiedGate.test_simplified_gate(
                self.model, self.val_loader, self.device
            )
            self._print_experiment_e_summary()
        except Exception as e:
            print(f"FAILED Exp E: {e}")
            import traceback
            traceback.print_exc()
            self.results['E_simplified_gate'] = {'error': str(e)}
        
        print("\n[Exp F] Embedding stability / alignment")
        print("-"*80)
        try:
            print("    Extracting embeddings...")
            embeddings = ExperimentF_EmbeddingStability.extract_embeddings(
                self.model, self.val_loader, self.device, max_batches=50
            )
            
            if embeddings is not None:
                print("    Computing embedding metrics...")
                metrics = ExperimentF_EmbeddingStability.compute_embedding_metrics(embeddings)
                
                print("    Saving visualization...")
                vis_path = os.path.join(self.save_dir, 'embedding_alignment.png')
                ExperimentF_EmbeddingStability.visualize_embedding_alignment(embeddings, vis_path)
                
                self.results['F_embedding_stability'] = {
                    'metrics': metrics,
                    'visualization': vis_path
                }
                self._print_experiment_f_summary()
            else:
                print("FAILED: could not extract embeddings")
                self.results['F_embedding_stability'] = {'error': 'Failed to extract embeddings'}
        except Exception as e:
            print(f"FAILED Exp F: {e}")
            import traceback
            traceback.print_exc()
            self.results['F_embedding_stability'] = {'error': str(e)}
        
        self.save_results()
        self.generate_report()
        
        print("\n"+"="*80)
        print("Phase-2 done. Results saved to:", self.save_dir)
        print("="*80)
    
    def _print_experiment_d_summary(self):
        """Print Experiment D summary."""
        if 'error' in self.results['D_random_gates']:
            return
        
        d = self.results['D_random_gates']
        print("\n  Results:")
        print(f"    Adaptive:           {d['adaptive']:.2f}%")
        print(f"    Fixed All-On:       {d['fixed_all_on']:.2f}% ({d['analysis']['adaptive_vs_all_on']:+.2f}%)")
        print(f"    Fixed Random (avg): {d['fixed_random_mean']:.2f}% ({d['analysis']['adaptive_vs_random_mean']:+.2f}%)")
        print(f"    Fixed Random range: [{d['fixed_random_min']:.2f}%, {d['fixed_random_max']:.2f}%]")
        print(f"\n  Conclusions:")
        for conclusion in d['analysis']['conclusion']:
            print(f"    {conclusion}")
    
    def _print_experiment_e_summary(self):
        """Print Experiment E summary."""
        if 'error' in self.results['E_simplified_gate']:
            return
        
        e = self.results['E_simplified_gate']
        print("\n  Results:")
        print(f"    Per-Token Gate:     {e['per_token_acc']:.2f}%")
        print(f"    Per-Layer Gate:     {e['per_layer_fixed_acc']:.2f}% ({e['analysis']['performance_gap']:+.2f}%)")
        print(f"\n  Per-token pattern:")
        pattern = e['per_token_pattern']
        print(f"    Mean: {pattern['mean']:.4f}, Std: {pattern['std']:.4f}")
        print(f"    Entropy: {pattern['entropy']:.4f}")
        print(f"    Has Clear Pattern: {pattern['has_clear_pattern']}")
        print(f"\n  Conclusions:")
        for conclusion in e['analysis']['conclusion']:
            print(f"    {conclusion}")
    
    def _print_experiment_f_summary(self):
        """Print Experiment F summary."""
        if 'error' in self.results['F_embedding_stability']:
            return
        
        f = self.results['F_embedding_stability']['metrics']
        print("\n  Embedding metrics:")
        if 'alignment' in f:
            print(f"    Image-text alignment: {f['alignment']['mean']:.4f} ± {f['alignment']['std']:.4f}")
        if 'clustering' in f and 'error' not in f['clustering']:
            print(f"    Clustering (silhouette): {f['clustering']['avg_silhouette']:.4f}")
        if 'separation' in f:
            print(f"    Inter-class separation: {f['separation']['image_center_dist']:.4f}")

    def save_results(self):
        """Write advanced_experiment_results.json."""
        results_path = os.path.join(self.save_dir, 'advanced_experiment_results.json')

        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_native(item) for item in obj)
            else:
                return obj

        serializable_results = convert_to_native(self.results)

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nSaved: {results_path}")
    
    def generate_report(self):
        """Write advanced_diagnostic_report.txt (English)."""
        report_path = os.path.join(self.save_dir, 'advanced_diagnostic_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("AdaptiveBiDirMaPLe phase-2 diagnostic report\n")
            f.write("="*80 + "\n\n")
            
            if 'D_random_gates' in self.results and 'error' not in self.results['D_random_gates']:
                f.write("[Exp D] Fixed vs random gates\n")
                f.write("-"*80 + "\n")
                
                d = self.results['D_random_gates']
                f.write(f"Accuracy:\n")
                f.write(f"  Adaptive:           {d['adaptive']:.2f}%\n")
                f.write(f"  Fixed All-On:       {d['fixed_all_on']:.2f}% (gap {d['analysis']['adaptive_vs_all_on']:+.2f}%)\n")
                f.write(f"  Fixed Random (avg): {d['fixed_random_mean']:.2f}% (gap {d['analysis']['adaptive_vs_random_mean']:+.2f}%)\n")
                f.write(f"  Fixed Random range: [{d['fixed_random_min']:.2f}%, {d['fixed_random_max']:.2f}%]\n\n")
                
                if abs(d['analysis']['adaptive_vs_all_on']) < 1.0:
                    f.write("PASS: Adaptive ~ All-On — learned gates add little\n")
                    verdict_d = True
                elif d['fixed_all_on'] > d['adaptive']:
                    f.write("PASS: All-On > Adaptive — learned gates may hurt\n")
                    verdict_d = True
                else:
                    f.write("Adaptive beats All-On by {:.2f}%\n".format(d['analysis']['adaptive_vs_all_on']))
                    verdict_d = False
                
                if d['analysis']['random_covers_adaptive']:
                    f.write("PASS: Random trials cover adaptive accuracy\n")
                    verdict_d = True
                elif abs(d['analysis']['adaptive_vs_random_mean']) < 1.0:
                    f.write("PASS: Adaptive ~ random mean\n")
                    verdict_d = True
                
                f.write(f"\nExp D verdict: {'supported' if verdict_d else 'not supported'}\n")
                if verdict_d:
                    f.write("  - Fixed schedules match adaptive; consider dropping adaptive gates\n\n")
                else:
                    f.write("  - Adaptive may be learning something useful\n\n")
            
            if 'E_simplified_gate' in self.results and 'error' not in self.results['E_simplified_gate']:
                f.write("[Exp E] Per-layer vs per-token gate\n")
                f.write("-"*80 + "\n")
                
                e = self.results['E_simplified_gate']
                f.write(f"Accuracy:\n")
                f.write(f"  Per-Token Gate: {e['per_token_acc']:.2f}%\n")
                f.write(f"  Per-Layer Gate: {e['per_layer_fixed_acc']:.2f}% (gap {e['analysis']['performance_gap']:+.2f}%)\n\n")
                
                pattern = e['per_token_pattern']
                f.write(f"Per-token pattern:\n")
                f.write(f"  values: {pattern['values']}\n")
                f.write(f"  mean: {pattern['mean']:.4f}, std: {pattern['std']:.4f}\n")
                f.write(f"  entropy: {pattern['entropy']:.4f}\n")
                f.write(f"  monotonicity: {pattern['monotonicity']:.4f}\n\n")
                
                verdict_e = False
                if e['analysis']['per_layer_sufficient']:
                    f.write("PASS: Per-layer sufficient; per-token may be overly complex\n")
                    verdict_e = True
                
                if e['analysis']['token_level_redundant']:
                    f.write("PASS: Per-token gates nearly uniform\n")
                    verdict_e = True
                
                f.write(f"\nExp E verdict: {'supported' if verdict_e else 'not supported'}\n")
                if verdict_e:
                    f.write("  - Prefer simpler per-layer gates\n\n")
                else:
                    f.write("  - Per-token gates may be justified\n\n")
            
            if 'F_embedding_stability' in self.results and 'error' not in self.results['F_embedding_stability']:
                f.write("[Exp F] Embedding stability / alignment\n")
                f.write("-"*80 + "\n")
                
                metrics = self.results['F_embedding_stability']['metrics']
                
                if 'alignment' in metrics:
                    f.write(f"Image-text alignment:\n")
                    f.write(f"  mean: {metrics['alignment']['mean']:.4f}\n")
                    f.write(f"  std: {metrics['alignment']['std']:.4f}\n")
                    f.write(f"  range: [{metrics['alignment']['min']:.4f}, {metrics['alignment']['max']:.4f}]\n\n")
                
                if 'clustering' in metrics and 'error' not in metrics['clustering']:
                    f.write(f"Clustering (silhouette):\n")
                    f.write(f"  image: {metrics['clustering']['image_silhouette']:.4f}\n")
                    f.write(f"  text:  {metrics['clustering']['text_silhouette']:.4f}\n")
                    f.write(f"  avg:   {metrics['clustering']['avg_silhouette']:.4f}\n\n")
                
                if 'separation' in metrics:
                    f.write(f"Inter-class separation:\n")
                    f.write(f"  image class centers: {metrics['separation']['image_center_dist']:.4f}\n")
                    f.write(f"  text class centers:  {metrics['separation']['text_center_dist']:.4f}\n\n")
                
                f.write("Exp F: compare these numbers to MaPLe / BiDirMaPLe baselines.\n")
                f.write("If adaptive runs show worse alignment or clustering, gates may hurt the CLIP prior.\n\n")
            
            f.write("="*80 + "\n")
            f.write("Phase-2 summary\n")
            f.write("="*80 + "\n\n")
            
            verified = []
            if 'D_random_gates' in self.results and 'error' not in self.results['D_random_gates']:
                d = self.results['D_random_gates']
                if abs(d['analysis']['adaptive_vs_all_on']) < 1.0 or d['analysis']['random_covers_adaptive']:
                    verified.append('D')
            
            if 'E_simplified_gate' in self.results and 'error' not in self.results['E_simplified_gate']:
                e = self.results['E_simplified_gate']
                if e['analysis']['per_layer_sufficient'] or e['analysis']['token_level_redundant']:
                    verified.append('E')
            
            f.write(f"Experiments with support: {len(verified)}/2 (D, E)\n\n")
            
            if 'D' in verified:
                f.write("[D] Learned gates redundant:\n")
                f.write("  - Fixed all-on or random can match adaptive\n\n")
            
            if 'E' in verified:
                f.write("[E] Per-token redundancy:\n")
                f.write("  - Little structured pattern at token level\n\n")
            
            f.write("Suggestions:\n")
            f.write("1. Drop adaptive gates if D/E pass — use fixed MaPLe / BiDirMaPLe.\n")
            f.write("2. If keeping adaptation: one global depth, discrete search, or meta-learned config.\n")
            f.write("3. Alternatives: MaPLe (e.g. n_ctx=4, depth=9), BiDirMaPLe, grid search.\n")
        
        print(f"Report saved: {report_path}")


class CrossModelComparison:
    """
    Compare embedding stability across AdaptiveBiDirMaPLe, MaPLe, BiDirMaPLe checkpoints.
    """

    @staticmethod
    def compare_models_across_epochs(models_dict, data_loader, device, save_dir):
        """
        Extract embeddings for each named checkpoint and compare drift / quality.

        Args:
            models_dict: name -> model, e.g. adaptive_epoch_0, maple, ...
        """
        os.makedirs(save_dir, exist_ok=True)

        results = {}

        for model_name, model in models_dict.items():
            print(f"  Processing {model_name}...")
            embeddings = ExperimentF_EmbeddingStability.extract_embeddings(
                model, data_loader, device, max_batches=50
            )

            if embeddings is not None:
                metrics = ExperimentF_EmbeddingStability.compute_embedding_metrics(embeddings)
                results[model_name] = {
                    'embeddings': embeddings,
                    'metrics': metrics
                }

        drift_results = {}
        for model_type in ['adaptive', 'maple', 'bidir']:
            if f'{model_type}_epoch_0' in results:
                base_emb = results[f'{model_type}_epoch_0']['embeddings']

                for epoch in [1, 4, 5, 9]:
                    key = f'{model_type}_epoch_{epoch}'
                    if key in results:
                        current_emb = results[key]['embeddings']
                        drift = ExperimentF_EmbeddingStability.compute_embedding_drift(
                            base_emb, current_emb
                        )
                        drift_results[key] = drift

        CrossModelComparison.visualize_drift_comparison(drift_results, save_dir)
        CrossModelComparison.visualize_quality_comparison(results, save_dir)

        return {
            'embeddings': results,
            'drift': drift_results
        }

    @staticmethod
    def visualize_drift_comparison(drift_results, save_dir):
        """Plot and log embedding drift across checkpoints."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        models = {}
        for key, drift in drift_results.items():
            model_type = key.split('_epoch_')[0]
            epoch = int(key.split('_epoch_')[1])

            if model_type not in models:
                models[model_type] = {'epochs': [], 'drifts': []}

            models[model_type]['epochs'].append(epoch)
            models[model_type]['drifts'].append(drift['avg_drift'])

        print("\nEmbedding drift by model:")
        print("-" * 50)
        for model_type, data in models.items():
            sorted_idx = np.argsort(data['epochs'])
            epochs = np.array(data['epochs'])[sorted_idx]
            drifts = np.array(data['drifts'])[sorted_idx]

            print(f"{model_type}:")
            for epoch, drift_val in zip(epochs, drifts):
                print(f"  Epoch {epoch}: {drift_val:.6f}")
            print()

            ax1.plot(epochs, drifts, marker='o', label=model_type)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Avg Embedding Drift')
        ax1.set_title('Embedding Drift over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        final_drifts = {model: data['drifts'][-1] for model, data in models.items()}
        ax2.bar(final_drifts.keys(), final_drifts.values())
        ax2.set_ylabel('Final Drift')
        ax2.set_title('Final Embedding Drift Comparison')
        ax2.grid(True, alpha=0.3)

        for i, (model, value) in enumerate(final_drifts.items()):
            ax2.text(i, value + 0.001, f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'drift_comparison.png'), dpi=150)
        plt.close()

        with open(os.path.join(save_dir, 'drift_values.txt'), 'w') as f:
            f.write("Embedding drift by model:\n")
            f.write("=" * 50 + "\n")
            for model_type, data in models.items():
                f.write(f"{model_type}:\n")
                for epoch, drift_val in zip(data['epochs'], data['drifts']):
                    f.write(f"  Epoch {epoch}: {drift_val:.6f}\n")
                f.write("\n")

    @staticmethod
    def visualize_quality_comparison(results, save_dir):
        """Bar charts for alignment / clustering / separation; print metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics_to_plot = ['alignment', 'clustering', 'separation']

        print("\nEmbedding quality by model:")
        print("-" * 80)

        all_metrics_data = {}

        for idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[idx]

            model_names = []
            values = []
            metric_data = {}

            for model_name, data in results.items():
                metrics = data['metrics']

                if metric_name == 'alignment' and metric_name in metrics:
                    model_names.append(model_name)
                    value = metrics['alignment']['mean']
                    values.append(value)
                    metric_data[model_name] = value
                    print(f"{model_name} - alignment: {value:.6f}")

                elif metric_name == 'clustering' and metric_name in metrics and 'error' not in metrics['clustering']:
                    model_names.append(model_name)
                    value = metrics['clustering']['avg_silhouette']
                    values.append(value)
                    metric_data[model_name] = value
                    print(f"{model_name} - clustering: {value:.6f}")

                elif metric_name == 'separation' and metric_name in metrics:
                    model_names.append(model_name)
                    value = metrics['separation']['image_center_dist']
                    values.append(value)
                    metric_data[model_name] = value
                    print(f"{model_name} - separation: {value:.6f}")

            all_metrics_data[metric_name] = metric_data

            if values:
                bars = ax.bar(range(len(model_names)), values)
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.set_title(metric_name.capitalize())
                ax.grid(True, alpha=0.3)

                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                            f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'quality_comparison.png'), dpi=150)
        plt.close()

        with open(os.path.join(save_dir, 'quality_values.txt'), 'w') as f:
            f.write("Embedding quality by model:\n")
            f.write("=" * 80 + "\n")

            for metric_name, metric_data in all_metrics_data.items():
                f.write(f"\n{metric_name.capitalize()}:\n")
                f.write("-" * 40 + "\n")
                for model_name, value in metric_data.items():
                    f.write(f"{model_name}: {value:.6f}\n")


def run_advanced_diagnostics(trainer, epoch=None):
    """
    Run phase-2 diagnostics from a trainer.

    Args:
        trainer: Trainer with model, loaders, device, cfg.
        epoch: Label for output folder (optional).
    """
    save_dir = os.path.join(
        trainer.cfg.OUTPUT_DIR, 
        f"advanced_diagnostics_epoch_{epoch if epoch else 'final'}"
    )
    
    runner = AdvancedExperimentRunner(
        model=trainer.model,
        train_loader=trainer.train_loader_x,
        val_loader=trainer.val_loader,
        device=trainer.device,
        save_dir=save_dir
    )
    
    runner.run_all_experiments()
    
    return runner.results


if __name__ == "__main__":
    print(__doc__)