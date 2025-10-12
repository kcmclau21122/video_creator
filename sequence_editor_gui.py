#!/usr/bin/env python3
"""
AI Video Creator - Interactive Sequence Editor
A Streamlit-based GUI for previewing and editing video sequences before rendering
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
import io

# Import existing modules
try:
    from config import Config
    from media_ingester import MediaItem
    from scene_analyzer import SceneAnalysis
    from sequencing_engine import SequenceItem
    from video_composer import VideoComposer
except ImportError:
    st.error("Could not import required modules. Make sure you're running from the project directory.")
    st.stop()


class SequenceEditor:
    """Manages sequence editing operations"""
    
    def __init__(self, sequence_path: Path):
        self.sequence_path = sequence_path
        self.sequence_data = None
        self.modified = False
        
    def load_sequence(self) -> Optional[Dict]:
        """Load sequence from JSON file"""
        if not self.sequence_path.exists():
            return None
        
        with open(self.sequence_path, 'r') as f:
            self.sequence_data = json.load(f)
        return self.sequence_data
    
    def save_sequence(self):
        """Save modified sequence back to JSON"""
        if self.sequence_data:
            # Backup original
            backup_path = self.sequence_path.with_suffix('.json.backup')
            if self.sequence_path.exists():
                import shutil
                shutil.copy2(self.sequence_path, backup_path)
            
            # Save updated sequence
            with open(self.sequence_path, 'w') as f:
                json.dump(self.sequence_data, f, indent=2)
            
            self.modified = False
            return True
        return False
    
    def get_items(self) -> List[Dict]:
        """Get list of sequence items"""
        if self.sequence_data and 'items' in self.sequence_data:
            return self.sequence_data['items']
        return []
    
    def remove_item(self, index: int):
        """Remove item at given index"""
        if self.sequence_data and 'items' in self.sequence_data:
            items = self.sequence_data['items']
            if 0 <= index < len(items):
                items.pop(index)
                # Reindex remaining items
                for i, item in enumerate(items):
                    item['sequence_index'] = i
                
                # Update metadata
                self.sequence_data['total_items'] = len(items)
                self._recalculate_timings()
                self.modified = True
    
    def move_item(self, from_index: int, to_index: int):
        """Move item from one position to another"""
        if self.sequence_data and 'items' in self.sequence_data:
            items = self.sequence_data['items']
            if 0 <= from_index < len(items) and 0 <= to_index < len(items):
                # Move the item
                item = items.pop(from_index)
                items.insert(to_index, item)
                
                # Reindex all items
                for i, item in enumerate(items):
                    item['sequence_index'] = i
                
                self._recalculate_timings()
                self.modified = True
    
    def _recalculate_timings(self):
        """Recalculate start times and total duration"""
        if not self.sequence_data or 'items' not in self.sequence_data:
            return
        
        items = self.sequence_data['items']
        current_time = 0.0
        
        for item in items:
            item['start_time'] = current_time
            current_time += item['duration']
        
        # Update total duration
        self.sequence_data['total_duration'] = current_time
        self.sequence_data['total_duration_minutes'] = current_time / 60.0


def get_thumbnail(media_path: str, max_size: tuple = (300, 300)) -> Image.Image:
    """Get thumbnail for media file"""
    path = Path(media_path)
    
    if not path.exists():
        # Create placeholder image
        img = Image.new('RGB', max_size, color='gray')
        return img
    
    try:
        # For images
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic', '.heif']:
            img = Image.open(path)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img
        
        # For videos - extract first frame
        elif path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(path))
            frame = clip.get_frame(0)
            clip.close()
            img = Image.fromarray(frame)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img
    except Exception as e:
        st.warning(f"Could not load thumbnail for {path.name}: {e}")
        img = Image.new('RGB', max_size, color='gray')
        return img


def render_item_card(item: Dict, index: int, total_items: int) -> tuple:
    """Render a media item card and return user actions"""
    col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
    
    with col1:
        # Display thumbnail
        try:
            thumb = get_thumbnail(item['media_item_path'])
            st.image(thumb, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading: {e}")
    
    with col2:
        # Item details
        media_path = Path(item['media_item_path'])
        st.markdown(f"**{index + 1}. {media_path.name}**")
        st.caption(f"{item.get('caption', 'No caption')}")
        st.caption(f"🎭 {item.get('scene_type', 'unknown')} | 🎨 {item.get('mood', 'neutral')}")
    
    with col3:
        # Timing info
        st.metric("Duration", f"{item['duration']:.1f}s")
        st.caption(f"Start: {item['start_time']:.1f}s")
    
    with col4:
        # Action buttons
        actions = {}
        
        # Move up
        if index > 0:
            if st.button("⬆️", key=f"up_{index}", help="Move up"):
                actions['move_up'] = True
        
        # Move down
        if index < total_items - 1:
            if st.button("⬇️", key=f"down_{index}", help="Move down"):
                actions['move_down'] = True
        
        # Remove
        if st.button("🗑️", key=f"remove_{index}", help="Remove"):
            actions['remove'] = True
    
    return actions


def main():
    """Main Streamlit application"""
    
    # Page config
    st.set_page_config(
        page_title="AI Video Sequence Editor",
        page_icon="🎬",
        layout="wide"
    )
    
    # Title and description
    st.title("🎬 AI Video Sequence Editor")
    st.markdown("Preview and edit your video sequence before rendering")
    
    # Initialize session state
    if 'editor' not in st.session_state:
        sequence_path = Config.OUTPUT_DIR / "video_sequence.json"
        st.session_state.editor = SequenceEditor(sequence_path)
        st.session_state.editor.load_sequence()
    
    editor = st.session_state.editor
    
    # Check if sequence exists
    if not editor.sequence_data:
        st.error("❌ No sequence found!")
        st.info("Run `python process_all_media.py` first to create a sequence.")
        return
    
    # Sidebar - Sequence info and controls
    with st.sidebar:
        st.header("📊 Sequence Info")
        
        items = editor.get_items()
        total_duration = editor.sequence_data.get('total_duration', 0)
        
        st.metric("Total Items", len(items))
        st.metric("Total Duration", f"{total_duration/60:.2f} min")
        st.metric("Avg Duration", f"{total_duration/len(items) if items else 0:.1f}s")
        
        if editor.modified:
            st.warning("⚠️ Unsaved changes!")
        
        st.divider()
        
        # Save button
        if st.button("💾 Save Changes", type="primary", disabled=not editor.modified, use_container_width=True):
            if editor.save_sequence():
                st.success("✅ Sequence saved!")
                st.rerun()
        
        # Reload button
        if st.button("🔄 Reload Original", use_container_width=True):
            editor.load_sequence()
            st.success("✅ Reloaded!")
            st.rerun()
        
        st.divider()
        
        # Render video button
        st.header("🎥 Render Video")
        
        render_ready = len(items) > 0
        
        if not render_ready:
            st.warning("Add items to sequence first")
        
        if editor.modified:
            st.warning("⚠️ Save changes before rendering")
        
        render_disabled = not render_ready or editor.modified
        
        if st.button("🎬 Start Rendering", 
                    type="primary", 
                    disabled=render_disabled,
                    use_container_width=True):
            st.session_state.start_render = True
    
    # Main content area
    st.header(f"📹 Video Sequence ({len(items)} items)")
    
    # Display items
    if not items:
        st.info("No items in sequence")
        return
    
    # Process any pending actions
    actions_to_process = []
    
    for idx, item in enumerate(items):
        st.divider()
        actions = render_item_card(item, idx, len(items))
        
        if actions:
            actions_to_process.append((idx, actions))
    
    # Process actions (do this after rendering to avoid rerun issues)
    if actions_to_process:
        for idx, actions in actions_to_process:
            if 'move_up' in actions:
                editor.move_item(idx, idx - 1)
                st.rerun()
            elif 'move_down' in actions:
                editor.move_item(idx, idx + 1)
                st.rerun()
            elif 'remove' in actions:
                if st.session_state.get(f'confirm_remove_{idx}'):
                    editor.remove_item(idx)
                    st.session_state[f'confirm_remove_{idx}'] = False
                    st.rerun()
                else:
                    st.session_state[f'confirm_remove_{idx}'] = True
                    st.warning(f"⚠️ Click remove again to confirm deletion of item {idx + 1}")
                    st.rerun()
    
    # Handle video rendering
    if st.session_state.get('start_render'):
        st.divider()
        st.header("🎥 Rendering Video")
        
        with st.spinner("Rendering video... This may take several minutes..."):
            try:
                # Run the video composition script
                result = subprocess.run(
                    ["python", "test_video_composition.py"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    st.success("✅ Video rendered successfully!")
                    
                    output_video = Config.OUTPUT_DIR / "final_video.mp4"
                    if output_video.exists():
                        st.video(str(output_video))
                        st.balloons()
                else:
                    st.error(f"❌ Rendering failed:\n{result.stderr}")
            
            except Exception as e:
                st.error(f"❌ Error during rendering: {e}")
        
        st.session_state.start_render = False


if __name__ == "__main__":
    main()