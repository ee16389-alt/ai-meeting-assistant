const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  draft: {
    save:  (data) => ipcRenderer.invoke('draft:save', data),
    load:  ()     => ipcRenderer.invoke('draft:load'),
    clear: ()     => ipcRenderer.invoke('draft:clear'),
  },
});
